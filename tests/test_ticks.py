"""Tests for `order_mgmt.ticks`."""

from __future__ import annotations

import numpy as np
import pytest

from order_mgmt.ticks import infer_tick, lookup_tick, resolve_tick


def test_lookup_known_markets() -> None:
    assert lookup_tick("GCM24") == 0.10
    assert lookup_tick("NQH20") == 0.25
    assert lookup_tick("ESH20") == 0.25
    assert lookup_tick("BPM20") == 0.0001
    assert lookup_tick("JYU24") == 0.000001
    assert lookup_tick("RXM25") == 0.01
    assert lookup_tick("VGH22") == 1.0
    assert lookup_tick("HOF22") == 0.0001


def test_lookup_unknown_returns_none() -> None:
    assert lookup_tick("XX99") is None
    assert lookup_tick("") is None
    assert lookup_tick("12345") is None


def test_resolve_falls_back_to_heuristic() -> None:
    assert resolve_tick("XX99", 0.5) == 0.5
    assert resolve_tick("GCM24", 0.5) == 0.10  # spec wins over fallback


def test_infer_tick_recovers_grid() -> None:
    coarse = 100.0 + np.arange(11) * 0.10
    assert infer_tick(coarse) == pytest.approx(0.10)
    fine = 1.0 + np.arange(20) * 0.0001
    assert infer_tick(fine) == pytest.approx(0.0001)


def test_infer_tick_ignores_float_noise_and_duplicates() -> None:
    # Repeated levels and out-of-order input still yield the smallest positive gap.
    prices = [100.0, 100.10, 100.10, 100.20, 100.0, 100.30]
    assert infer_tick(prices) == pytest.approx(0.10)


def test_infer_tick_needs_two_distinct_prices() -> None:
    with pytest.raises(ValueError):
        infer_tick([5.0, 5.0, 5.0])
