"""EWMA / EWMV recursion tests vs a brute-force re-application of the documented formula.

Exercises the team's `ewma_ewmv` in `src/regime.py`.

NOTE: `_brute_force` below intentionally mirrors the *source loop*, so these tests
prove only that the code matches its own docstring — they cannot catch deviations
from the spec. For a spec-derived (Algorithm 1) oracle that does, see
tests/test_ewma_spec.py.
"""

from __future__ import annotations

import numpy as np

from regime import ewma_ewmv


def _brute_force(eta: np.ndarray, half_life: int) -> tuple[np.ndarray, np.ndarray]:
    """Re-apply the documented recursion step by step.

    Mirrors the source loop (NOT the spec) — so it reproduces the same eta[1] skip.
    A genuinely spec-faithful oracle lives in tests/test_ewma_spec.py.
    """
    n = len(eta)
    out_ewma = np.full(n, np.nan)
    out_ewmv = np.full(n, np.nan)
    if n < 3:
        return out_ewma, out_ewmv
    lam = 2.0 ** (-1.0 / half_life)
    # j = 2 init
    sumW = 1.0
    sumWX = float(eta[0])
    sumWSS = 0.0  # (eta[0] - ewma)^2 = 0 since ewma = eta[0]
    for j in range(3, n + 1):
        x = float(eta[j - 1])
        sumW = lam * sumW + 1.0
        sumWX = lam * sumWX + x
        ewma = sumWX / sumW
        sumWSS = lam * sumWSS + (x - ewma) ** 2
        ewmv = (sumWSS / sumW) ** 0.5
        out_ewma[j - 1] = ewma
        out_ewmv[j - 1] = ewmv
    return out_ewma, out_ewmv


def test_ewma_matches_documented_recursion() -> None:
    rng = np.random.default_rng(123)
    eta = rng.standard_normal(50)
    actual_ewma, actual_ewmv = ewma_ewmv(eta, half_life=10)
    expected_ewma, expected_ewmv = _brute_force(eta, half_life=10)
    np.testing.assert_allclose(actual_ewma[2:], expected_ewma[2:], rtol=1e-12)
    np.testing.assert_allclose(actual_ewmv[2:], expected_ewmv[2:], rtol=1e-12)
    assert np.isnan(actual_ewma[0]) and np.isnan(actual_ewma[1])
    assert np.isnan(actual_ewmv[0]) and np.isnan(actual_ewmv[1])


def test_ewma_constant_input_converges_to_constant() -> None:
    eta = np.full(500, 7.0)
    ewma, ewmv = ewma_ewmv(eta, half_life=20)
    np.testing.assert_allclose(ewma[-1], 7.0, atol=1e-9)
    np.testing.assert_allclose(ewmv[-1], 0.0, atol=1e-9)


def test_ewmv_is_nonnegative() -> None:
    rng = np.random.default_rng(7)
    eta = rng.standard_normal(200)
    _, ewmv = ewma_ewmv(eta, half_life=15)
    assert np.all(ewmv[2:] >= 0)


def test_short_half_life_tracks_recent_values_faster() -> None:
    """Step input: fast EWMA catches up sooner than slow one."""
    eta = np.concatenate([np.zeros(50), np.ones(50)])
    ewma_fast, _ = ewma_ewmv(eta, half_life=2)
    ewma_slow, _ = ewma_ewmv(eta, half_life=100)
    # 25 bars after the step: fast should be near 1, slow far from 1
    assert ewma_fast[74] > 0.9
    assert ewma_slow[74] < 0.5
