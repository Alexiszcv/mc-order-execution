"""Inverse-transform / composition sampling reproduces the source empirical PMF."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from order_mgmt.mc.samplers import build_cdf, sample_composition, sample_from_epdf


def test_build_cdf_monotone_ends_at_one() -> None:
    support, cumprob = build_cdf(Counter({0: 10, 2: 30, 5: 60}))
    assert list(support) == [0, 2, 5]
    assert np.all(np.diff(cumprob) >= 0)
    assert cumprob[-1] == 1.0


def test_inverse_transform_reproduces_pmf() -> None:
    epdf = Counter({0: 10, 1: 25, 2: 40, 3: 15, 5: 10})  # note the gap at 4
    total = sum(epdf.values())
    rng = np.random.default_rng(0)
    draws = sample_from_epdf(epdf, 200_000, rng=rng)

    for ell, c in epdf.items():
        emp = np.mean(draws == ell)
        assert abs(emp - c / total) < 0.01, f"freq mismatch at {ell}"
    assert not np.any(draws == 4)  # gap is never sampled


def test_single_bucket() -> None:
    rng = np.random.default_rng(1)
    draws = sample_from_epdf(Counter({3: 7}), 1000, rng=rng)
    assert np.all(draws == 3)


def test_empty_epdf_raises() -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        sample_from_epdf(Counter(), 10, rng=rng)


def test_composition_respects_cell_weights() -> None:
    counts_RU = {(1, 1, 1): Counter({1: 100}), (2, 2, 2): Counter({9: 100})}
    counts_RD = {(1, 1, 1): Counter({1: 100}), (2, 2, 2): Counter({9: 100})}
    weights = {(1, 1, 1): 0.25, (2, 2, 2): 0.75}
    rng = np.random.default_rng(3)
    r_u, _ = sample_composition(counts_RU, counts_RD, 100_000, cell_weights=weights, rng=rng)
    # cell (1,1,1) emits 1, cell (2,2,2) emits 9 -> share of 9s ~ 0.75
    share_high = np.mean(r_u == 9)
    assert abs(share_high - 0.75) < 0.01
