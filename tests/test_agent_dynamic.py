"""Tests for `order_mgmt.agent.dynamic` — adaptive + DP schedules and forward simulation.

Hand-computed schedules pin the optimal-stopping math; a forward-sim test pins the
fill/market mechanics. A data-backed test confirms the end-to-end comparison runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from order_mgmt.agent.dynamic import (
    adaptive_schedule,
    dp_schedule,
    simulate_dynamic,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_OHLC = ROOT / "data" / "Gold" / "GCM24.csv"
GOLD_AGENT = ROOT / "data" / "Gold" / "AIAgent_Gold.csv"

# Per-cell horizon samples for a buy: favorable = R_D, adverse = R_U.
CD = {
    "RD": {
        1: [3, 3, 3, 2, 2, 2, 1, 1, 1, 1],  # survival: ≥1=1.0, ≥2=0.6, ≥3=0.3
        2: [5, 5, 5, 4, 4, 4, 3, 3, 2, 2],  # survival: ≥4=0.6, ≥5=0.3
    },
    "RU": {1: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]},  # adverse mean w = 0.5
}


def test_adaptive_schedule_grows_with_horizon() -> None:
    # target 0.5: r=1 → largest ℓ with q(1,ℓ)≥0.5 is 2; r=2 → 4. More time ⇒ post wider.
    sched = adaptive_schedule(CD, "buy", target=0.5, tau_max=2)
    assert sched == {1: 2, 2: 4}


def test_dp_schedule_shrinks_toward_deadline() -> None:
    # Backward induction (w=0.5): optimal ℓ is 1 with one minute left, 2 with two left.
    sched = dp_schedule(CD, "buy", tau_max=2)
    assert sched == {1: 1, 2: 2}


def test_simulate_dynamic_fills_first_minute() -> None:
    # sched {1:1, 2:2}; buy arrival 100, tick 1. Bar 0 open 100, low 97 ≤ 100-2 → fill @98.
    o = np.array([100.0, 100.0])
    h = np.array([100.0, 100.0])
    low = np.array([97.0, 99.0])
    c = np.array([100.0, 100.0])
    short, minute = simulate_dynamic("buy", {1: 1, 2: 2}, o, h, low, c, 100.0, 1.0, 2)
    assert short == pytest.approx(2.0)  # bought 2 ticks under arrival
    assert minute == 0


def test_simulate_dynamic_markets_at_deadline() -> None:
    # Never reaches the limit → market at the last close (100.5) → buy shortfall -0.5.
    o = np.array([100.0, 100.0])
    h = np.array([100.0, 100.0])
    low = np.array([99.6, 99.6])
    c = np.array([100.0, 100.5])
    short, minute = simulate_dynamic("buy", {1: 1, 2: 2}, o, h, low, c, 100.0, 1.0, 2)
    assert short == pytest.approx(-0.5)
    assert minute == -1


@pytest.mark.skipif(
    not (GOLD_OHLC.exists() and GOLD_AGENT.exists()), reason="Gold data not present"
)
def test_evaluate_agent_dynamic_runs() -> None:
    from order_mgmt.agent.dynamic import evaluate_agent_dynamic
    from order_mgmt.agent.loader import load_agent_series, load_market_for_agent

    agent = load_agent_series(GOLD_AGENT, market="Gold")
    df_ohlcv, tick, proper_days, _ = load_market_for_agent(GOLD_OHLC.parent)
    res = evaluate_agent_dynamic(
        agent, df_ohlcv, tick=tick, proper_days=proper_days, tau=5, half_life=20,
        M=3, N=3, K=3, fill_rate_target=0.6, j_start=200,
    )
    for name in ("single", "blend", "cutoff", "adaptive", "dp"):
        assert res[name]["shortfall"].size > 0
    # all schemes evaluated on the same decision set
    sizes = {res[n]["shortfall"].size for n in ("single", "adaptive", "dp")}
    assert len(sizes) == 1
