"""Tests for `order_mgmt.pipeline.load_market_indexed`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from order_mgmt.pipeline import load_market_indexed


def _write_csv(path: Path, rows: list[tuple]) -> None:
    path.write_text("\n".join(f"{t},{o},{h},{lo},{c},{v}" for t, o, h, lo, c, v in rows))


def test_load_market_indexed_returns_indexed_ohlcv(tmp_path: Path) -> None:
    rows = [
        ("2022.11.07.09:30:00", 100.0, 100.5, 99.8, 100.3, 100),
        ("2022.11.07.09:31:00", 100.3, 101.0, 100.0, 100.7, 120),
    ]
    # Need enough bars to clear the 90% min_fraction filter — make a "full" day
    full_day_rows = [
        (f"2022.11.07.{9 + i // 60:02d}:{i % 60:02d}:00", 100.0, 100.5, 99.8, 100.3, 100)
        for i in range(30, 30 + 480)
    ]
    _write_csv(tmp_path / "TEST22.csv", full_day_rows)
    df = load_market_indexed(tmp_path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "time"
    assert {"open", "high", "low", "close", "volume", "contract"}.issubset(df.columns)
    assert len(df) == 480


def test_load_market_indexed_empty_dir(tmp_path: Path) -> None:
    df = load_market_indexed(tmp_path)
    assert df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
