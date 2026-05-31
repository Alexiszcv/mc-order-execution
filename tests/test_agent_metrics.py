"""Tests for `order_mgmt.agent.metrics`.

A fully hand-computed scenario pins the fill + shortfall arithmetic; pure unit
tests cover the per-decision window helper. A data-backed Gold test checks the
end-to-end path runs and is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from order_mgmt.agent.loader import AgentSeries, load_agent_series
from order_mgmt.agent.metrics import (
    DecisionFill,
    _window_stats,
    evaluate_agent_execution,
    size_weighted_shortfall,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_OHLC = ROOT / "data" / "Gold" / "GCM24.csv"
GOLD_AGENT = ROOT / "data" / "Gold" / "AIAgent_Gold.csv"


# --- _window_stats (pure) ----------------------------------------------------


def _ns(ts: str) -> int:
    return int(pd.Timestamp(ts).value)


def test_window_stats_empty_interval_returns_none() -> None:
    idx_ns = np.array([_ns("2024-01-02 00:00"), _ns("2024-01-02 00:05")], dtype=np.int64)
    z = np.array([1.0, 1.0])
    # Interval [00:01, 00:02) contains no bars.
    out = _window_stats(idx_ns, z, z, z, z, z, z, _ns("2024-01-02 00:01"), int(6e10))
    assert out is None


def test_window_stats_ohlc_and_vwap() -> None:
    idx_ns = np.array(
        [_ns("2024-01-02 00:00"), _ns("2024-01-02 00:01"), _ns("2024-01-02 00:02")],
        dtype=np.int64,
    )
    opens = np.array([100.0, 101.0, 102.0])
    highs = np.array([100.0, 103.0, 102.0])
    lows = np.array([99.0, 101.0, 100.0])
    closes = np.array([100.0, 102.0, 101.0])
    typ = (highs + lows + closes) / 3.0
    vol = np.array([1.0, 1.0, 2.0])
    tau_ns = int(pd.Timedelta(minutes=3).value)
    o, h, low, c, vwap = _window_stats(
        idx_ns, opens, highs, lows, closes, typ, vol, _ns("2024-01-02 00:00"), tau_ns
    )
    assert o == 100.0  # first bar's open
    assert h == 103.0  # max high
    assert low == 99.0  # min low
    assert c == 101.0  # last bar's close
    expected_vwap = float((typ * vol).sum() / vol.sum())
    assert vwap == pytest.approx(expected_vwap)


# --- sizing rules (pure) ------------------------------------------------------


def _fill(shortfall: float, qty: int, fill_prob: float, ell_star: int, vol_proxy: float) -> DecisionFill:
    return DecisionFill(
        t=pd.Timestamp("2024-01-02"), side="buy", qty=qty, arrival_price=0.0, open_j=0.0,
        realized_price=0.0, filled=True, ell_star=ell_star, slippage_ticks=0.0,
        shortfall_ticks=shortfall, fill_prob=fill_prob, vol_proxy=vol_proxy,
    )


def test_size_weighted_shortfall_each_rule() -> None:
    fills = [
        _fill(+2.0, qty=1, fill_prob=0.5, ell_star=4, vol_proxy=10.0),
        _fill(-4.0, qty=3, fill_prob=1.0, ell_star=2, vol_proxy=5.0),
    ]
    # agent weights [1,3] -> (2 - 12)/4 = -2.5
    assert size_weighted_shortfall(fills, "agent") == pytest.approx(-2.5)
    # confidence weights fill_prob·ℓ* = [2,2] -> (4 - 8)/4 = -1.0
    assert size_weighted_shortfall(fills, "confidence") == pytest.approx(-1.0)
    # inverse-vol weights [0.1,0.2] -> (0.2 - 0.8)/0.3 = -2.0
    assert size_weighted_shortfall(fills, "inverse_vol") == pytest.approx(-2.0)
    assert size_weighted_shortfall([], "agent") == 0.0


# --- hand-computed end-to-end -------------------------------------------------


def _flat_ohlc(n_bars: int = 21) -> tuple[pd.DataFrame, list]:
    """A single day of identical bars: open=100, high=102, low=100, close=101, vol=10.

    With tick=1 each bar (τ=1 window) has R_U=2, R_D=0, range=2, Δx=0.
    """
    idx = pd.date_range("2024-01-02 00:00", periods=n_bars, freq="1min", name="time")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 102.0,
            "low": 100.0,
            "close": 101.0,
            "volume": 10.0,
        },
        index=idx,
    )
    return df, [pd.Timestamp("2024-01-02")]


def test_evaluate_handcomputed_sell_fill() -> None:
    df, proper_days = _flat_ohlc()

    # One OOS sell at 00:15 with arrival price 100. Single regime cell (M=N=K=1).
    agent_df = pd.DataFrame(
        {"price": [100.0], "position": [-1.0], "dpos": [-1.0], "side": ["sell"], "qty": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 00:15:00")], name="time"),
    )
    agent = AgentSeries(market="T", df=agent_df)

    res = evaluate_agent_execution(
        agent,
        df,
        tick=1.0,
        proper_days=proper_days,
        tau=1,
        half_life=2,
        M=1,
        N=1,
        K=1,
        fill_rate_target=0.6,
        j_start=2,
        train_end=pd.Timestamp("2024-01-02 00:10:00"),
    )

    assert res.n_decisions == 1
    assert res.n_filled == 1
    assert res.fill_rate == 1.0
    f = res.fills[0]
    # ePDF for sells is {R_U=2}, so ℓ* = 2; R_U_j = 2 >= 2 → fills at open+2 = 102.
    assert f.ell_star == 2
    assert f.filled is True
    assert f.realized_price == pytest.approx(102.0)
    # Sell shortfall vs arrival 100 = (102-100)/1 = +2 ticks (beat arrival).
    assert f.shortfall_ticks == pytest.approx(2.0)
    assert f.slippage_ticks == pytest.approx(2.0)  # vs window open 100
    assert res.mean_shortfall_ticks == pytest.approx(2.0)
    assert res.median_shortfall_ticks == pytest.approx(2.0)
    # market-on-decision reference is 0 → value-add equals our mean shortfall.
    assert res.value_add_vs_market_ticks == pytest.approx(res.mean_shortfall_ticks)
    # captured improvement = shortfall·tick·qty = 2·1·1.
    assert res.captured_improvement_notional == pytest.approx(2.0)
    # VWAP of the bar = (102+100+101)/3 = 101 → vwap shortfall +1 → value-add = 2-1 = 1.
    assert res.value_add_vs_vwap_ticks == pytest.approx(1.0)
    assert res.unfilled_tail_cost_ticks == 0.0  # nothing unfilled


@pytest.mark.skipif(
    not (GOLD_OHLC.exists() and GOLD_AGENT.exists()), reason="Gold data not present"
)
def test_gold_end_to_end_runs_and_is_deterministic() -> None:
    from order_mgmt.agent.loader import load_market_for_agent

    agent = load_agent_series(GOLD_AGENT, market="Gold")
    df_ohlcv, tick, proper_days, _ = load_market_for_agent(GOLD_OHLC.parent)

    kw = dict(
        tick=tick,
        proper_days=proper_days,
        tau=5,
        half_life=20,
        M=3,
        N=3,
        K=3,
        fill_rate_target=0.6,
        j_start=200,
    )
    r1 = evaluate_agent_execution(agent, df_ohlcv, **kw)
    r2 = evaluate_agent_execution(agent, df_ohlcv, **kw)

    assert r1.n_decisions > 0
    assert 0.0 <= r1.fill_rate <= 1.0
    assert np.isfinite(r1.mean_shortfall_ticks)
    assert np.isfinite(r1.median_shortfall_ticks)
    # market-on-decision identity holds on real data too.
    assert r1.value_add_vs_market_ticks == pytest.approx(r1.mean_shortfall_ticks)
    # Determinism: identical inputs → identical headline numbers.
    assert r2.n_decisions == r1.n_decisions
    assert r2.mean_shortfall_ticks == pytest.approx(r1.mean_shortfall_ticks)
    assert r2.median_shortfall_ticks == pytest.approx(r1.median_shortfall_ticks)
