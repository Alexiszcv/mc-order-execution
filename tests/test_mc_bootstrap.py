"""Phase A bootstrap: point-estimate correctness + basic-CI coverage on known truth."""

from __future__ import annotations

from collections import Counter

import numpy as np

from order_mgmt.mc.bootstrap import _basic_ci, _bootstrap_mean, bootstrap_strategy


def test_fill_rate_point_matches_survival() -> None:
    epdf = Counter({0: 20, 1: 30, 2: 30, 3: 20})  # total 100
    # target 0.5 -> ell* = 2 (P(R>=2) = 0.5); fill rate = 0.5
    res = bootstrap_strategy(epdf, side="sell", fill_rate_target=0.5, n_boot=500, seed=0)
    assert res.ell_star == 2
    assert abs(res.fill_rate.mean - 0.5) < 1e-9
    assert res.fill_rate.ci_low <= 0.5 <= res.fill_rate.ci_high
    assert res.mse_fill_rate > 0.0


def test_slippage_bootstrapped_when_provided() -> None:
    epdf = Counter({0: 50, 4: 50})
    slips = [3.0] * 30 + [-1.0] * 20  # mean (90 - 20)/50 = 1.4
    res = bootstrap_strategy(
        epdf, side="sell", fill_rate_target=0.5, n_boot=400, seed=1, slippage_samples=slips
    )
    assert abs(res.avg_slippage_ticks.mean - 1.4) < 1e-9
    assert res.avg_slippage_ticks.ci_low <= 1.4 <= res.avg_slippage_ticks.ci_high


def test_empty_epdf_degenerate() -> None:
    res = bootstrap_strategy(Counter(), side="buy", fill_rate_target=0.6, n_boot=100, seed=0)
    assert res.fill_rate.n == 0
    assert res.mse_fill_rate == 0.0


def test_basic_ci_coverage_on_known_proportion() -> None:
    """Across many synthetic datasets, the 95% basic CI covers the true mean ~95% of the time."""
    p_true = 0.6
    n_obs, n_trials, n_boot = 400, 300, 300
    rng = np.random.default_rng(12345)
    covered = 0
    for _ in range(n_trials):
        data = rng.binomial(1, p_true, n_obs).astype(float)
        point = float(data.mean())
        reps = _bootstrap_mean(data, n_boot, rng)
        lo, hi, _ = _basic_ci(point, reps, 0.95)
        covered += int(lo <= p_true <= hi)
    coverage = covered / n_trials
    assert 0.88 <= coverage <= 0.99, f"coverage {coverage:.3f} off nominal 0.95"
