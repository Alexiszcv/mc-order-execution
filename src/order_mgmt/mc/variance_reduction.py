"""Phase C — variance-reduction techniques applied to the MC estimators.

Each function returns a `VRResult` reporting the estimate and the variance-reduction ratio
(plain estimator variance / reduced estimator variance; >1 means it helped). The four
cheat-sheet techniques:

- **antithetic** — pair each BM path Z with −Z; slippage is monotone in the path level, so
  the pair has negative covariance and the paired-mean estimator has lower variance.
- **control variate** — the driftless-BM close offset has known mean 0 and correlates with
  slippage; subtract β·(C − E[C]).
- **conditional MC** — the regime conditioning is E[fill|cell]; integrating the within-cell
  Bernoulli analytically removes that variance component (law of total variance).
- **importance sampling** — exponentially tilt the empirical range PMF toward the rare
  large ranges (the chase tail) to estimate a tail probability with less variance.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from order_mgmt.strategy import pick_ell_star

from .paths import _ranges_from_path
from .results import MCEstimate, Side, VRResult
from .simulator import _fill_and_slippage


def _clt_estimate(samples: np.ndarray, ci: float) -> MCEstimate:
    return MCEstimate.from_samples(samples, ci=ci, method="clt")


def _gbm_slip(z, sigma_bar, n_sub, side, ell_star):
    """slippage, close, path-mean for BM increments `z` (shape (n, n_steps))."""
    sigma_step = sigma_bar / math.sqrt(n_sub)
    path = np.cumsum(sigma_step * z, axis=1)
    r_u, r_d = _ranges_from_path(path, round_ticks=True)
    close = path[:, -1]
    realized = r_u if side == "sell" else r_d
    _, slip = _fill_and_slippage(realized, close, side, ell_star)
    return slip, close, path.mean(axis=1)


def antithetic_slippage(
    decision_epdf: Counter,
    *,
    side: Side,
    fill_rate_target: float,
    n_paths: int,
    tau: int,
    sigma_bar: float,
    n_sub: int = 10,
    ci: float = 0.95,
    seed: int = 0,
) -> VRResult:
    """Antithetic-variate estimate of mean slippage (gbm model) vs plain MC of equal budget."""
    rng = np.random.default_rng(seed)
    ell_star = pick_ell_star(decision_epdf, fill_rate_target)
    n_steps = tau * n_sub
    n_pairs = max(n_paths // 2, 1)

    z = rng.standard_normal((n_pairs, n_steps))
    slip_pos, _, _ = _gbm_slip(z, sigma_bar, n_sub, side, ell_star)
    slip_neg, _, _ = _gbm_slip(-z, sigma_bar, n_sub, side, ell_star)
    paired = 0.5 * (slip_pos + slip_neg)
    reduced_var = float(paired.var(ddof=1) / n_pairs)

    z_plain = rng.standard_normal((2 * n_pairs, n_steps))
    slip_plain, _, _ = _gbm_slip(z_plain, sigma_bar, n_sub, side, ell_star)
    baseline_var = float(slip_plain.var(ddof=1) / (2 * n_pairs))

    ratio = baseline_var / reduced_var if reduced_var > 0 else float("inf")
    est = MCEstimate(
        mean=float(paired.mean()),
        std_error=math.sqrt(reduced_var),
        ci_low=float(paired.mean()) - 1.96 * math.sqrt(reduced_var),
        ci_high=float(paired.mean()) + 1.96 * math.sqrt(reduced_var),
        n=n_pairs,
        method="clt",
    )
    return VRResult("antithetic", est, baseline_var, reduced_var, ratio, 2 * n_pairs, seed)


def control_variate_slippage(
    decision_epdf: Counter,
    *,
    side: Side,
    fill_rate_target: float,
    n_paths: int,
    tau: int,
    sigma_bar: float,
    control: str = "twap",
    n_sub: int = 10,
    ci: float = 0.95,
    seed: int = 0,
) -> VRResult:
    """Control-variate estimate of mean slippage. Control has known mean 0 (driftless BM).

    control="twap": close offset (close − open); control="vwap": mean path level. Both have
    expectation 0 under the driftless model and correlate with slippage.
    """
    rng = np.random.default_rng(seed)
    ell_star = pick_ell_star(decision_epdf, fill_rate_target)
    n_steps = tau * n_sub
    z = rng.standard_normal((n_paths, n_steps))
    slip, close, path_mean = _gbm_slip(z, sigma_bar, n_sub, side, ell_star)

    c = close if control == "twap" else path_mean  # E[C] = 0
    var_c = float(c.var(ddof=1))
    beta = float(np.cov(slip, c, ddof=1)[0, 1] / var_c) if var_c > 0 else 0.0
    y_cv = slip - beta * (c - 0.0)

    baseline_var = float(slip.var(ddof=1) / n_paths)
    reduced_var = float(y_cv.var(ddof=1) / n_paths)
    ratio = baseline_var / reduced_var if reduced_var > 0 else float("inf")
    est = _clt_estimate(y_cv, ci)
    return VRResult(
        f"control:{control}", est, baseline_var, reduced_var, ratio, n_paths, seed
    )


def conditional_mc_fill_rate(
    counts_decision: dict,
    cell_weights: dict,
    *,
    fill_rate_target: float,
    n_paths: int = 10_000,
    seed: int = 0,
) -> VRResult:
    """Conditional-MC marginal fill rate: Σ_c w_c · P(R >= ℓ*_c), read analytically.

    Removes the within-cell Bernoulli variance (law of total variance). Variances are
    reported for a nominal budget of `n_paths` paths: plain ~ p(1−p)/n; conditional ~
    Var_w(p_c)/n.
    """
    cells = [c for c, w in cell_weights.items() if w > 0 and counts_decision.get(c)]
    if not cells:
        raise ValueError("no usable cells")
    w = np.array([cell_weights[c] for c in cells], dtype=float)
    w /= w.sum()

    p_c = np.empty(len(cells))
    for i, c in enumerate(cells):
        counter = counts_decision[c]
        ell_star = pick_ell_star(counter, fill_rate_target)
        total = sum(counter.values())
        p_c[i] = sum(cnt for ell, cnt in counter.items() if ell >= ell_star) / total

    p = float((w * p_c).sum())
    between_var = float((w * (p_c - p) ** 2).sum())  # Var_w(p_c)
    baseline_var = p * (1.0 - p) / n_paths
    reduced_var = between_var / n_paths
    ratio = baseline_var / reduced_var if reduced_var > 0 else float("inf")
    se = math.sqrt(reduced_var)
    est = MCEstimate(p, se, p - 1.96 * se, p + 1.96 * se, n_paths, "clt")
    return VRResult("conditional", est, baseline_var, reduced_var, ratio, n_paths, seed)


def _pmf_from_counter(epdf: Counter) -> tuple[np.ndarray, np.ndarray]:
    support = np.array(sorted(epdf.keys()), dtype=np.int64)
    p = np.array([epdf[int(k)] for k in support], dtype=float)
    return support, p / p.sum()


def _tilt_to_mean(support: np.ndarray, p: np.ndarray, target_mean: float) -> float:
    """Find λ so the exponentially-tilted PMF has mean ~ target_mean (bisection, λ>=0)."""
    base_mean = float((support * p).sum())
    if target_mean <= base_mean:
        return 0.0

    def tilted_mean(lam: float) -> float:
        w = np.exp(lam * support) * p
        w /= w.sum()
        return float((support * w).sum())

    lo, hi = 0.0, 1.0
    while tilted_mean(hi) < target_mean and hi < 50.0:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tilted_mean(mid) < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def importance_sampling_chase_tail(
    decision_epdf: Counter,
    *,
    tail_quantile: float = 0.05,
    tilt: float | None = None,
    n_paths: int = 20_000,
    ci: float = 0.95,
    seed: int = 0,
) -> VRResult:
    """IS estimate of a rare large-range tail probability P(R >= a), with exponential tilting.

    `a` is the smallest tick value whose empirical survival <= `tail_quantile` (the rare
    large ranges that drive the costly chase). `tilt` λ defaults to the value that recenters
    the sampling PMF on `a` (near-optimal); the estimator reweights by f/g.
    """
    rng = np.random.default_rng(seed)
    support, p = _pmf_from_counter(decision_epdf)
    surv = np.cumsum(p[::-1])[::-1]  # surv[i] = P(R >= support[i])
    tail_idx = np.where(surv <= tail_quantile)[0]
    a = int(support[tail_idx[0]]) if tail_idx.size else int(support[-1])

    # plain MC
    plain = rng.choice(support, size=n_paths, p=p)
    ind_plain = (plain >= a).astype(float)
    baseline_var = float(ind_plain.var(ddof=1) / n_paths)

    # importance sampling under tilted g
    lam = _tilt_to_mean(support, p, float(a)) if tilt is None else float(tilt)
    g = np.exp(lam * support) * p
    g /= g.sum()
    draws_idx = rng.choice(len(support), size=n_paths, p=g)
    drawn = support[draws_idx]
    weight = p[draws_idx] / g[draws_idx]
    is_vals = (drawn >= a).astype(float) * weight
    reduced_var = float(is_vals.var(ddof=1) / n_paths)
    ratio = baseline_var / reduced_var if reduced_var > 0 else float("inf")

    mean = float(is_vals.mean())
    se = math.sqrt(reduced_var)
    est = MCEstimate(mean, se, mean - 1.96 * se, mean + 1.96 * se, n_paths, "clt")
    return VRResult("importance", est, baseline_var, reduced_var, ratio, n_paths, seed)


__all__ = [
    "antithetic_slippage",
    "conditional_mc_fill_rate",
    "control_variate_slippage",
    "importance_sampling_chase_tail",
]


# expose theta for tests/diagnostics via a thin helper
def tail_probability(decision_epdf: Counter, tail_quantile: float = 0.05) -> tuple[int, float]:
    support, p = _pmf_from_counter(decision_epdf)
    surv = np.cumsum(p[::-1])[::-1]
    tail_idx = np.where(surv <= tail_quantile)[0]
    a = int(support[tail_idx[0]]) if tail_idx.size else int(support[-1])
    return a, float(p[support >= a].sum())
