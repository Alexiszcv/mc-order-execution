"""Tests for `order_mgmt.agent.slicing` and the multi-scheme evaluator.

Hand-computed fills for each scheme, plus a cross-check that the `single` scheme in
`evaluate_agent_schemes` reproduces `evaluate_agent_execution` exactly (guards the
duplicated no-lookahead loop from drifting).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from order_mgmt.agent.slicing import (
    fill_blend,
    fill_capped,
    fill_cutoff,
    fill_single,
    fill_time_slice,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_OHLC = ROOT / "data" / "Gold" / "GCM24.csv"
GOLD_AGENT = ROOT / "data" / "Gold" / "AIAgent_Gold.csv"


# A 4-bar buy window, open=100, tick=1. A dip to 98 in bar 1 (R_D=2), recovers to close 101.
WO = np.array([100.0, 100.0, 99.0, 100.0])
WH = np.array([100.0, 100.0, 100.0, 101.0])
WL = np.array([100.0, 98.0, 99.0, 100.0])
WC = np.array([100.0, 99.0, 100.0, 101.0])


def test_fill_single_buy_fills_at_limit() -> None:
    # ℓ*=2: R_D over the window = open-min = 100-98 = 2 ≥ 2 → fills at open-2 = 98.
    price, frac = fill_single("buy", 2, WO, WH, WL, WC, 1.0)
    assert price == pytest.approx(98.0)
    assert frac == 1.0


def test_fill_single_buy_chases_when_unreached() -> None:
    # ℓ*=3: R_D = 2 < 3 → no fill → chase at close 101.
    price, frac = fill_single("buy", 3, WO, WH, WL, WC, 1.0)
    assert price == pytest.approx(101.0)
    assert frac == 0.0


def test_fill_blend_mixes_open_and_limit() -> None:
    # f=0.5: half at open (100), half at the limit fill (98) → 99.
    price, frac = fill_blend("buy", 2, WO, WH, WL, WC, 1.0, f=0.5)
    assert price == pytest.approx(99.0)
    assert frac == pytest.approx(1.0)  # (1-f) certain + f·filled = 0.5 + 0.5
    # if the limit misses, the open half is still certain → fill_frac = 0.5.
    _, frac_miss = fill_blend("buy", 9, WO, WH, WL, WC, 1.0, f=0.5)
    assert frac_miss == pytest.approx(0.5)


def test_fill_cutoff_caps_chase_midwindow() -> None:
    # ℓ*=3 never reached (min dip 2). cutoff_frac=0.5 of 4 bars = bar index 1 → market at C[1]=99,
    # which beats the single-shot chase at the close (101).
    price, frac = fill_cutoff("buy", 3, WO, WH, WL, WC, 1.0, cutoff_frac=0.5)
    assert price == pytest.approx(99.0)
    assert frac == 0.0
    # ℓ*=2 IS reached at bar 1 (low 98 ≤ 98) → fills at the limit 98.
    price2, frac2 = fill_cutoff("buy", 2, WO, WH, WL, WC, 1.0, cutoff_frac=0.5)
    assert price2 == pytest.approx(98.0)
    assert frac2 == 1.0


def test_fill_time_slice_scales_limit_and_averages() -> None:
    # k=2 children. ℓ_sub = round(2/√2) = round(1.414) = 1.
    # child A = bars[0:2] (open 100, low 98 → R_D=2 ≥ 1) → fills at 99.
    # child B = bars[2:4] (open 99, low 99 → R_D=0 < 1) → chases at close 101.
    # parent = mean(99, 101) = 100; fill_fraction = 1/2.
    price, frac = fill_time_slice("buy", 2, WO, WH, WL, WC, 1.0, k=2)
    assert price == pytest.approx(100.0)
    assert frac == pytest.approx(0.5)


def test_fill_capped_fills_or_stops() -> None:
    # ℓ*=2 fills at the limit 98 (bar 1 low 98 ≤ 98) before any stop.
    price, frac = fill_capped("buy", 2, WO, WH, WL, WC, 1.0, cap=8)
    assert price == pytest.approx(98.0)
    assert frac == 1.0
    # ℓ*=9 never fills; cap=1 → stop when price rises to open+1 = 101 (bar 3 high 101).
    price2, frac2 = fill_capped("buy", 9, WO, WH, WL, WC, 1.0, cap=1)
    assert price2 == pytest.approx(101.0)  # loss capped at 1 tick vs the open
    assert frac2 == 0.0


@pytest.mark.skipif(
    not (GOLD_OHLC.exists() and GOLD_AGENT.exists()), reason="Gold data not present"
)
def test_single_scheme_matches_evaluate_agent_execution() -> None:
    from order_mgmt.agent.loader import load_agent_series, load_market_for_agent
    from order_mgmt.agent.metrics import evaluate_agent_execution, evaluate_agent_schemes

    agent = load_agent_series(GOLD_AGENT, market="Gold")
    df_ohlcv, tick, proper_days, _ = load_market_for_agent(GOLD_OHLC.parent)
    kw = dict(
        tick=tick, proper_days=proper_days, tau=5, half_life=20, M=3, N=3, K=3,
        fill_rate_target=0.6, j_start=200,
    )
    ref = evaluate_agent_execution(agent, df_ohlcv, **kw)
    multi = evaluate_agent_schemes(agent, df_ohlcv, **kw)

    ref_short = np.array([f.shortfall_ticks for f in ref.fills])
    got_short = multi["single"]["shortfall"]
    assert got_short.shape == ref_short.shape
    np.testing.assert_allclose(got_short, ref_short, atol=1e-9)
