"""Parametric fitting: parameter recovery, AIC family selection, survival, sampling."""

from __future__ import annotations

from collections import Counter

import numpy as np

from order_mgmt.mc.fit import fit_all_regimes, fit_distribution


def _geometric_zero_based(p: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n values from a geometric on {0,1,2,...} with success prob p."""
    return rng.geometric(p, size=n) - 1  # rng.geometric is on {1,2,...}


def test_geometric_param_recovery() -> None:
    rng = np.random.default_rng(0)
    obs = _geometric_zero_based(0.3, 50_000, rng)
    fit = fit_distribution(obs, families=("geometric",))
    assert fit.family == "geometric"
    assert abs(fit.params[0] - 0.3) < 0.02


def test_aic_selects_true_family() -> None:
    rng = np.random.default_rng(1)
    obs = _geometric_zero_based(0.35, 40_000, rng)
    fit = fit_distribution(obs)  # all families compete
    assert fit.family == "geometric", f"expected geometric, got {fit.family} (aic {fit.aic:.1f})"


def test_survival_monotone_and_bounded() -> None:
    rng = np.random.default_rng(2)
    obs = _geometric_zero_based(0.25, 20_000, rng)
    fit = fit_distribution(obs)
    surv = [fit.survival(ell) for ell in range(0, 30)]
    assert abs(surv[0] - 1.0) < 1e-9          # P(R >= 0) = 1
    assert all(0.0 <= s <= 1.0 for s in surv)
    assert all(surv[i] >= surv[i + 1] - 1e-12 for i in range(len(surv) - 1))


def test_sampling_reproduces_fit() -> None:
    rng = np.random.default_rng(3)
    obs = _geometric_zero_based(0.3, 20_000, rng)
    fit = fit_distribution(obs)
    draws = fit.sample(200_000, rng=rng)
    sup = np.asarray(fit.support)
    emp = np.bincount(draws, minlength=sup.size).astype(float)[: sup.size]
    emp /= emp.sum()
    assert np.max(np.abs(emp - np.asarray(fit.pmf))) < 0.01


def test_fit_accepts_counter() -> None:
    fit = fit_distribution(Counter({0: 200, 1: 150, 2: 90, 3: 40, 4: 20}))
    assert fit.n_obs == 500
    assert abs(sum(fit.pmf) - 1.0) < 1e-9


def test_fit_all_regimes_skips_sparse() -> None:
    counts = {
        (1, 1, 1): Counter({0: 100, 1: 80, 2: 40, 3: 10}),
        (2, 2, 2): Counter({5: 1}),  # too few -> skipped
        (3, 3, 3): Counter(),         # empty -> skipped
    }
    fits = fit_all_regimes(counts)
    assert (1, 1, 1) in fits
    assert (2, 2, 2) not in fits
    assert (3, 3, 3) not in fits
