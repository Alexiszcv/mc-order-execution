"""Tests for `order_mgmt.agent.loader`.

Covers the two load-bearing inferences: Excel-serial-day decoding and trade
direction from the position change (NOT from the price change).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from order_mgmt.agent.loader import (
    EXCEL_EPOCH,
    decode_agent_timestamps,
    load_agent_series,
    trade_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_AGENT_CSV = ROOT / "data" / "Gold" / "AIAgent_Gold.csv"


def test_excel_epoch_is_1899_12_30() -> None:
    assert pd.Timestamp("1899-12-30") == EXCEL_EPOCH


def test_decode_timestamps_hand_checked() -> None:
    # 45293 -> 2024-01-02; +0h +5m -> 00:05:00.
    raw = pd.DataFrame({"day": [45293], "hour": [0], "minute": [5]})
    idx = decode_agent_timestamps(raw)
    assert list(idx) == [pd.Timestamp("2024-01-02 00:05:00")]
    assert idx.name == "time"


def _write_raw(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "AIAgent_Test.csv"
    p.write_text("\n".join(rows) + "\n")
    return p


def test_direction_from_position_not_price(tmp_path: Path) -> None:
    # Columns: day, hour, minute, price, position(extra).
    # Row 2: position 2->5 (buy) while price 100.5->100.0 FALLS.
    # Row 3: position 5->-1 (sell) while price 100.0->100.1 RISES.
    # If direction came from price-change sign these would flip — they must not.
    path = _write_raw(
        tmp_path,
        [
            "45293,0,0,100.0,0",
            "45293,0,5,100.5,2",
            "45293,0,10,100.0,5",
            "45293,0,15,100.1,-1",
        ],
    )
    series = load_agent_series(path, market="Test")
    df = series.df

    assert list(df["position"]) == [0.0, 2.0, 5.0, -1.0]
    # First row has no prior position -> no trade (NA under pandas' str dtype).
    assert pd.isna(df["side"].iloc[0])
    assert df["side"].iloc[1] == "buy"   # dpos +2
    assert df["side"].iloc[2] == "buy"   # dpos +3 despite price falling
    assert df["side"].iloc[3] == "sell"  # dpos -6 despite price rising
    assert list(df["qty"]) == [0, 2, 3, 6]


def test_trade_decisions_drops_holds(tmp_path: Path) -> None:
    path = _write_raw(
        tmp_path,
        [
            "45293,0,0,100.0,0",
            "45293,0,5,100.5,0",  # hold (dpos 0)
            "45293,0,10,100.0,3",  # buy 3
        ],
    )
    trades = trade_decisions(load_agent_series(path, market="Test"))
    assert len(trades) == 1
    assert trades["side"].iloc[0] == "buy"
    assert trades["qty"].iloc[0] == 3


@pytest.mark.skipif(not GOLD_AGENT_CSV.exists(), reason="Gold AIAgent data not present")
def test_gold_agent_loads_with_expected_shape() -> None:
    series = load_agent_series(GOLD_AGENT_CSV, market="Gold")
    df = series.df
    # Verified from the raw data during planning.
    assert df["position"].min() == -7.0
    assert df["position"].max() == 8.0
    trades = trade_decisions(series)
    assert len(trades) == 1276
    # All decision timestamps are inside the agent's coverage window.
    assert df.index.min() >= pd.Timestamp("2024-01-01")
    assert df.index.max() <= pd.Timestamp("2024-06-01")
