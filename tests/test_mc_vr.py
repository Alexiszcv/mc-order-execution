"""Phase C variance reduction: each technique reduces estimator variance (ratio > 1)."""

from __future__ import annotations

from collections import Counter

from order_mgmt.mc.variance_reduction import (
    antithetic_slippage,
    conditional_mc_fill_rate,
    control_variate_slippage,
    importance_sampling_chase_tail,
    tail_probability,
)

_EPDF = Counter({0: 30, 1: 25, 2: 20, 3: 12, 4: 8, 5: 5})


def test_antithetic_reduces_variance() -> None:
    vr = antithetic_slippage(
        _EPDF, side="sell", fill_rate_target=0.5, n_paths=20000, tau=5, sigma_bar=3.0, seed=0
    )
    assert vr.variance_reduction_ratio > 1.0


def test_control_variate_reduces_variance() -> None:
    vr = control_variate_slippage(
        _EPDF, side="sell", fill_rate_target=0.5, n_paths=20000, tau=5,
        sigma_bar=3.0, control="twap", seed=1,
    )
    assert vr.variance_reduction_ratio > 1.0


def test_conditional_mc_matches_analytic_marginal() -> None:
    counts = {
        (1, 1, 1): Counter({0: 50, 5: 50}),  # ell*=5 -> p_c = 0.5
        (2, 2, 2): Counter({0: 90, 5: 10}),  # ell*=0 -> p_c = 1.0
    }
    weights = {(1, 1, 1): 0.5, (2, 2, 2): 0.5}
    vr = conditional_mc_fill_rate(counts, weights, fill_rate_target=0.5, n_paths=10000)
    assert abs(vr.estimate.mean - 0.75) < 1e-9   # 0.5*0.5 + 0.5*1.0
    assert vr.variance_reduction_ratio > 1.0     # between-cell var < total var


def test_importance_sampling_tail() -> None:
    epdf = Counter({0: 40, 1: 25, 2: 15, 3: 10, 4: 6, 5: 3, 6: 1})
    _a, theta = tail_probability(epdf, tail_quantile=0.05)
    vr = importance_sampling_chase_tail(epdf, tail_quantile=0.05, n_paths=20000, seed=2)
    assert abs(vr.estimate.mean - theta) < 0.01
    assert vr.variance_reduction_ratio > 1.0
