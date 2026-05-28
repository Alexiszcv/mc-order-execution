"""Phase A — nonparametric bootstrap of the strategy's fill rate and slippage.

Implements the cheat-sheet Bootstrap Algorithm: treat a regime's R ePDF as the empirical
CDF F̃, resample with replacement, recompute the statistic, and form a confidence interval
+ an MSE estimate. Fill rate is bootstrapped from the ePDF directly (it depends only on
P(R >= ℓ*)); slippage is bootstrapped from the realized per-decision slippage list that the
backtest already produces, since the chase-on-unfill cost is not encoded in the R ePDF.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from order_mgmt.strategy import pick_ell_star

from .results import BootstrapResult, MCEstimate


def _expand(epdf: Counter) -> np.ndarray:
    """Counter {ℓ: count} -> flat array of observed ℓ values."""
    if not epdf:
        return np.empty(0, dtype=np.int64)
    vals = []
    for ell, c in epdf.items():
        vals.extend([int(ell)] * int(c))
    return np.array(vals, dtype=np.int64)


def _basic_ci(point: float, reps: np.ndarray, ci: float) -> tuple[float, float, float]:
    """Cheat-sheet basic bootstrap CI (2θ̂ − q_u, 2θ̂ − q_l) + bootstrap SE."""
    alpha = 1.0 - ci
    q_lo = float(np.quantile(reps, alpha / 2.0))
    q_hi = float(np.quantile(reps, 1.0 - alpha / 2.0))
    se = float(reps.std(ddof=1)) if reps.size >= 2 else 0.0
    return 2.0 * point - q_hi, 2.0 * point - q_lo, se


def _bootstrap_mean(data: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap replicates of the sample mean of `data`."""
    n = data.size
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[b] = data[idx].mean()
    return reps


def bootstrap_strategy(
    epdf_decision: Counter,
    *,
    side: str,
    fill_rate_target: float,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    slippage_samples: list[float] | None = None,
) -> BootstrapResult:
    """Bootstrap CIs/MSE for the fill rate (and, if given, the realized slippage) at ℓ*.

    `epdf_decision` is the regime's R_U (sell) or R_D (buy) Counter. ℓ* is chosen once on
    the full ePDF; the bootstrap quantifies the sampling uncertainty of the resulting fill
    rate. `slippage_samples` is the per-decision slippage list from a `BacktestResult` for
    this regime/side; pass it to also bootstrap the average slippage.
    """
    rng = np.random.default_rng(seed)
    ell_star = pick_ell_star(epdf_decision, fill_rate_target)

    obs = _expand(epdf_decision)
    if obs.size == 0:
        empty = MCEstimate(0.0, 0.0, 0.0, 0.0, 0, "basic")
        return BootstrapResult(side, ell_star, empty, empty, 0.0, 0.0, n_boot, seed)

    fill_ind = (obs >= ell_star).astype(float)
    point_fill = float(fill_ind.mean())
    fill_reps = _bootstrap_mean(fill_ind, n_boot, rng)
    lo, hi, se = _basic_ci(point_fill, fill_reps, ci)
    fill_est = MCEstimate(point_fill, se, lo, hi, n_boot, "basic")
    mse_fill = float(np.mean((fill_reps - point_fill) ** 2))

    if slippage_samples:
        slip = np.asarray(slippage_samples, dtype=float)
        point_slip = float(slip.mean())
        slip_reps = _bootstrap_mean(slip, n_boot, rng)
        s_lo, s_hi, s_se = _basic_ci(point_slip, slip_reps, ci)
        slip_est = MCEstimate(point_slip, s_se, s_lo, s_hi, n_boot, "basic")
        mse_slip = float(np.mean((slip_reps - point_slip) ** 2))
    else:
        slip_est = MCEstimate(0.0, 0.0, 0.0, 0.0, 0, "basic")
        mse_slip = 0.0

    return BootstrapResult(
        side=side,
        ell_star=ell_star,
        fill_rate=fill_est,
        avg_slippage_ticks=slip_est,
        mse_fill_rate=mse_fill,
        mse_slippage=mse_slip,
        n_boot=n_boot,
        seed=seed,
    )
