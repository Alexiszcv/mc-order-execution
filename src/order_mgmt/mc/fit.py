"""Phase B — fit a parametric distribution to a regime's range, per family, select by AIC.

Range = number of ticks ≥ 0. Every candidate family is turned into a **discrete PMF over
integer ticks** so the log-likelihood (and therefore AIC) is comparable across discrete and
continuous families: continuous families are fit on `obs + 0.5` (bin midpoints, avoids the
zero-density problem) and discretised with the [ℓ, ℓ+1) convention `P(ℓ)=F(ℓ+1)−F(ℓ)`.

Selection is by minimum AIC; the KS statistic vs the empirical CDF is reported as a
goodness-of-fit summary (its p-value is the asymptotic approximation — exact only for
continuous data, so treat it as indicative).

The half-normal family is included deliberately: it is the Brownian-motion range law, so it
is the analytic bridge to the GBM path model in `paths.py`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import stats

DEFAULT_FAMILIES = ("geometric", "nbinom", "gamma", "weibull", "halfnorm")


@dataclass(frozen=True)
class FitResult:
    """A fitted distribution over integer ticks, with provenance and goodness-of-fit.

    `support` is 0..ell_max and `pmf` the matching probabilities (sum 1). All accessors
    operate on these arrays, so sampling/survival need no scipy at call time.
    """

    family: str
    params: tuple[float, ...]
    aic: float
    ks_stat: float
    ks_pvalue: float
    n_obs: int
    support: tuple[int, ...]
    pmf: tuple[float, ...]

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.support, dtype=np.int64), np.asarray(self.pmf, dtype=float)

    def prob(self, ell: int) -> float:
        """P(R = ell)."""
        sup, pmf = self._arrays()
        if ell < sup[0] or ell > sup[-1]:
            return 0.0
        return float(pmf[ell - sup[0]])

    def cdf(self, ell: int) -> float:
        """P(R <= ell)."""
        sup, pmf = self._arrays()
        return float(pmf[sup <= ell].sum())

    def survival(self, ell: int) -> float:
        """P(R >= ell)."""
        sup, pmf = self._arrays()
        return float(pmf[sup >= ell].sum())

    def sample(self, n: int, *, rng: np.random.Generator) -> np.ndarray:
        """Inverse-transform draw of n integer-tick values from the fitted PMF."""
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        sup, pmf = self._arrays()
        cumprob = np.cumsum(pmf)
        cumprob[-1] = 1.0
        idx = np.searchsorted(cumprob, rng.random(n), side="left")
        return sup[np.clip(idx, 0, sup.size - 1)]

    def to_counter(self, scale: int = 1_000_000) -> Counter:
        """Render the fitted PMF as integer counts so `pick_ell_star` can consume it."""
        c: Counter = Counter()
        for ell, p in zip(self.support, self.pmf, strict=False):
            w = round(p * scale)
            if w > 0:
                c[ell] = w
        return c


def _empty_counter_to_array(values) -> np.ndarray:
    if isinstance(values, Counter):
        out = []
        for ell, c in values.items():
            out.extend([int(ell)] * int(c))
        return np.array(out, dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def _discretize(cdf, support: np.ndarray) -> np.ndarray:
    """[ℓ, ℓ+1) discretisation of a continuous CDF: P(ℓ) = F(ℓ+1) − F(ℓ), normalised."""
    edges = np.arange(support[0], support[-1] + 2, dtype=float)
    pmf = np.clip(np.diff(cdf(edges)), 0.0, None)
    total = pmf.sum()
    if total <= 0:
        raise ValueError("degenerate discretised pmf")
    return pmf / total


def _fit_family(family: str, obs: np.ndarray, support: np.ndarray) -> tuple[tuple, int, np.ndarray]:
    """Return (params, n_free_params, pmf-over-support) for one family. Raises on failure."""
    mean = float(obs.mean())
    var = float(obs.var(ddof=1)) if obs.size >= 2 else 0.0

    if family == "geometric":  # on {0,1,2,...}: p = 1/(1+mean)
        p = 1.0 / (1.0 + mean) if mean > 0 else 1.0
        pmf = np.power(1.0 - p, support) * p
        return (p,), 1, pmf / pmf.sum()

    if family == "nbinom":  # method of moments; only valid when overdispersed
        if var <= mean or mean <= 0:
            raise ValueError("nbinom needs var > mean > 0")
        p = mean / var
        r = mean * mean / (var - mean)
        pmf = stats.nbinom.pmf(support, r, p)
        s = pmf.sum()
        if s <= 0:
            raise ValueError("nbinom pmf degenerate")
        return (r, p), 2, pmf / s

    if family == "gamma":
        a, _loc, scale = stats.gamma.fit(obs + 0.5, floc=0)
        pmf = _discretize(lambda x: stats.gamma.cdf(x, a, scale=scale), support)
        return (a, scale), 2, pmf

    if family == "weibull":
        c, _loc, scale = stats.weibull_min.fit(obs + 0.5, floc=0)
        pmf = _discretize(lambda x: stats.weibull_min.cdf(x, c, scale=scale), support)
        return (c, scale), 2, pmf

    if family == "halfnorm":
        _loc, scale = stats.halfnorm.fit(obs + 0.5, floc=0)
        pmf = _discretize(lambda x: stats.halfnorm.cdf(x, scale=scale), support)
        return (scale,), 1, pmf

    raise ValueError(f"unknown family {family!r}")


def fit_distribution(values, *, families=DEFAULT_FAMILIES) -> FitResult:
    """Fit each candidate family to integer-tick `values` (array or Counter); return best by AIC."""
    obs = _empty_counter_to_array(values)
    if obs.size == 0:
        raise ValueError("cannot fit an empty sample")

    obs_max = int(obs.max())
    ell_max = min(max(obs_max * 3, obs_max + 30), 5000)
    support = np.arange(0, ell_max + 1, dtype=np.int64)

    # empirical CDF on the shared support (for KS)
    emp_pmf = np.bincount(obs, minlength=ell_max + 1).astype(float)[: ell_max + 1]
    emp_pmf /= emp_pmf.sum()
    emp_cdf = np.cumsum(emp_pmf)

    best: FitResult | None = None
    for family in families:
        try:
            params, k, pmf = _fit_family(family, obs, support)
        except Exception:
            continue
        logp = np.log(np.clip(pmf[obs], 1e-300, None))
        loglik = float(logp.sum())
        aic = 2.0 * k - 2.0 * loglik
        model_cdf = np.cumsum(pmf)
        ks_stat = float(np.max(np.abs(model_cdf - emp_cdf)))
        ks_pvalue = float(stats.kstwobign.sf(np.sqrt(obs.size) * ks_stat))
        cand = FitResult(
            family=family,
            params=tuple(float(x) for x in params),
            aic=aic,
            ks_stat=ks_stat,
            ks_pvalue=ks_pvalue,
            n_obs=int(obs.size),
            support=tuple(int(s) for s in support),
            pmf=tuple(float(x) for x in pmf),
        )
        if best is None or cand.aic < best.aic:
            best = cand

    if best is None:
        raise RuntimeError("all candidate families failed to fit")
    return best


def fit_all_regimes(counts: dict, *, families=DEFAULT_FAMILIES) -> dict:
    """Fit a distribution per regime cell. Cells with too few observations are skipped."""
    out: dict = {}
    for cell, counter in counts.items():
        if not counter or sum(counter.values()) < 2:
            continue
        try:
            out[cell] = fit_distribution(counter, families=families)
        except Exception:
            continue
    return out
