"""Tests for `order_mgmt.agent.benchmarks` — the baseline comparison harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLD_OHLC = ROOT / "data" / "Gold" / "GCM24.csv"
GOLD_AGENT = ROOT / "data" / "Gold" / "AIAgent_Gold.csv"


@pytest.mark.skipif(
    not (GOLD_OHLC.exists() and GOLD_AGENT.exists()), reason="Gold data not present"
)
def test_benchmarks_run_same_decision_set_and_market_has_thin_tail() -> None:
    from order_mgmt.agent.benchmarks import NAMES, evaluate_agent_benchmarks
    from order_mgmt.agent.loader import load_agent_series, load_market_for_agent

    agent = load_agent_series(GOLD_AGENT, market="Gold")
    df_ohlcv, tick, proper_days, _ = load_market_for_agent(GOLD_OHLC.parent)
    res = evaluate_agent_benchmarks(
        agent, df_ohlcv, tick=tick, proper_days=proper_days, tau=5, half_life=20,
        M=3, N=3, K=3, fill_rate_target=0.6, j_start=200, seed=0,
    )
    sizes = {res[nm]["shortfall"].size for nm in NAMES}
    assert len(sizes) == 1 and sizes.pop() > 0  # all strategies on the same decisions
    # market (all-in at the open) fills immediately, so its tail is far thinner than a
    # limit strategy that can miss and chase.
    market_p5 = np.percentile(res["market"]["shortfall"], 5)
    regime_p5 = np.percentile(res["regime"]["shortfall"], 5)
    assert market_p5 > regime_p5
