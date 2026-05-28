"""BM path models: reflection-principle mean, bridge survival law, sanity invariants."""

from __future__ import annotations

import math

import numpy as np

from order_mgmt.mc.paths import (
    halfnormal_range_mean,
    sigma_from_range_level,
    simulate_bridge_ranges,
    simulate_gbm_ranges,
)


def test_sigma_calibration_roundtrips() -> None:
    tau = 5
    e_r = 12.0
    sigma_bar = sigma_from_range_level(e_r, tau)
    # full range mean = sqrt(8/pi)*sigma_window should recover E[R]
    recon = sigma_bar * math.sqrt(tau) * math.sqrt(8.0 / math.pi)
    assert abs(recon - e_r) < 1e-9


def test_forward_max_matches_reflection_principle() -> None:
    tau = 5
    sigma_window = 40.0
    sigma_bar = sigma_window / math.sqrt(tau)
    rng = np.random.default_rng(0)
    r_u, _ = simulate_gbm_ranges(
        sigma_bar, tau, 4000, n_sub=400, round_ticks=False, rng=rng
    )
    expected = halfnormal_range_mean(sigma_bar, tau)  # ~31.9
    ratio = r_u.mean() / expected
    # discrete path slightly under-estimates the continuous max -> ratio a touch below 1
    assert 0.93 <= ratio <= 1.02, f"ratio {ratio:.3f}"


def test_bridge_max_survival_law() -> None:
    tau = 5
    sigma_window = 30.0
    sigma_bar = sigma_window / math.sqrt(tau)
    b, a = 5.0, 20.0
    rng = np.random.default_rng(1)
    r_u, _ = simulate_bridge_ranges(
        sigma_bar, tau, np.full(8000, b), n_sub=300, round_ticks=False, rng=rng
    )
    emp = float(np.mean(r_u >= a))
    analytic = math.exp(-2.0 * a * (a - b) / sigma_window**2)  # ~0.513
    assert abs(emp - analytic) < 0.04, f"emp {emp:.3f} vs analytic {analytic:.3f}"


def test_ranges_nonneg_and_monotone_in_sigma() -> None:
    tau = 5
    rng = np.random.default_rng(2)
    ru_lo, rd_lo = simulate_gbm_ranges(5.0 / math.sqrt(tau), tau, 4000, rng=rng)
    ru_hi, rd_hi = simulate_gbm_ranges(20.0 / math.sqrt(tau), tau, 4000, rng=rng)
    assert np.all(ru_lo >= 0) and np.all(rd_lo >= 0)
    assert (ru_hi + rd_hi).mean() > (ru_lo + rd_lo).mean()
