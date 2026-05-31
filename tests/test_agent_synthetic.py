"""Tests for `order_mgmt.agent.synthetic` — the zero-shot generator.

Determinism (same seed → identical series) and shape-compatibility (the synthetic
series feeds the real metrics path unchanged) are the two contracts that matter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from order_mgmt.agent.loader import trade_decisions
from order_mgmt.agent.metrics import evaluate_agent_execution
from order_mgmt.agent.synthetic import synth_agent_series

ROOT = Path(__file__).resolve().parents[1]
GOLD_OHLC = ROOT / "data" / "Gold" / "GCM24.csv"


def _toy_ohlc() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 00:00", periods=600, freq="1min", name="time")
    # mildly varying bars so regimes are non-degenerate
    base = 100.0 + (pd.Series(range(len(idx))) % 7) * 0.1
    return pd.DataFrame(
        {
            "open": base.to_numpy(),
            "high": (base + 0.3).to_numpy(),
            "low": (base - 0.2).to_numpy(),
            "close": (base + 0.1).to_numpy(),
            "volume": 10.0,
        },
        index=idx,
    )


def test_synth_has_agent_series_shape() -> None:
    s = synth_agent_series(_toy_ohlc(), seed=0, n_decisions=50)
    assert list(s.df.columns) == ["price", "position", "dpos", "side", "qty"]
    assert s.df.index.name == "time"
    assert s.df.index.is_monotonic_increasing
    assert s.df["position"].abs().max() <= 8
    # the random walk produces some trades
    assert len(trade_decisions(s)) > 0


def test_synth_deterministic_under_seed() -> None:
    df = _toy_ohlc()
    a = synth_agent_series(df, seed=42, n_decisions=50)
    b = synth_agent_series(df, seed=42, n_decisions=50)
    pd.testing.assert_frame_equal(a.df, b.df)
    c = synth_agent_series(df, seed=7, n_decisions=50)
    # different seed → different series (positions differ somewhere)
    assert not a.df["position"].equals(c.df["position"])


@pytest.mark.skipif(not GOLD_OHLC.exists(), reason="Gold data not present")
def test_synthetic_runs_through_metrics_on_gold_ohlc() -> None:
    from order_mgmt.agent.loader import load_market_for_agent

    df_ohlcv, tick, proper_days, _ = load_market_for_agent(GOLD_OHLC.parent)
    agent = synth_agent_series(df_ohlcv, seed=0, n_decisions=400)
    res = evaluate_agent_execution(
        agent,
        df_ohlcv,
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
    # Genericity: the exact same pipeline produces a sane result on never-seen data.
    assert res.n_decisions > 0
    assert 0.0 <= res.fill_rate <= 1.0
