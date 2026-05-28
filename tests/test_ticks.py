"""Tests for `order_mgmt.ticks`."""

from __future__ import annotations

from order_mgmt.ticks import lookup_tick, resolve_tick


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
