"""No-lookahead invariant tests: outputs at time j may not depend on data after j."""

from __future__ import annotations

import numpy as np

from regime import ewma_ewmv


def test_ewma_ignores_future_under_tail_shuffle() -> None:
    """Shuffling eta[j_split:] must not change ewma[:j_split]."""
    rng = np.random.default_rng(42)
    n, j_split = 200, 100

    eta1 = rng.standard_normal(n)
    eta2 = eta1.copy()
    rng.shuffle(eta2[j_split:])

    assert not np.array_equal(eta1, eta2)  # the shuffle actually changed the tail

    ewma1, ewmv1 = ewma_ewmv(eta1, half_life=20)
    ewma2, ewmv2 = ewma_ewmv(eta2, half_life=20)

    np.testing.assert_array_equal(ewma1[:j_split], ewma2[:j_split])
    np.testing.assert_array_equal(ewmv1[:j_split], ewmv2[:j_split])


def test_ewma_truncated_input_matches_prefix() -> None:
    """ewma(eta[:k]) must equal ewma(eta)[:k]."""
    rng = np.random.default_rng(99)
    eta = rng.standard_normal(150)
    k = 80
    full_ewma, full_ewmv = ewma_ewmv(eta, half_life=15)
    prefix_ewma, prefix_ewmv = ewma_ewmv(eta[:k], half_life=15)
    np.testing.assert_array_equal(full_ewma[:k], prefix_ewma)
    np.testing.assert_array_equal(full_ewmv[:k], prefix_ewmv)
