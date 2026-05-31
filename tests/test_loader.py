"""Tests for `order_mgmt.loader` — exercise the math, not the I/O."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from order_mgmt.loader import (
    MarketSpec,
    active_day_mask,
    drop_low_liquidity_days,
    load_contract,
    pick_active_contract,
)


def _ohlc_csv_text(rows: list[tuple[str, float, float, float, float, float]]) -> str:
    return "\n".join(f"{t},{o},{h},{lo},{c},{v}" for t, o, h, lo, c, v in rows)


def _synth_day(date_str: str, n_minutes: int, contract: str, vol: float = 10.0) -> pd.DataFrame:
    start = pd.Timestamp(f"{date_str} 09:30:00")
    times = pd.date_range(start, periods=n_minutes, freq="1min")
    return pd.DataFrame(
        {
            "time": times,
            "open": 1000.0,
            "high": 1001.0,
            "low": 999.0,
            "close": 1000.0,
            "volume": vol,
            "contract": contract,
        }
    )


def test_parses_ohlc_schema(tmp_path: Path) -> None:
    p = tmp_path / "TEST22.csv"
    p.write_text(
        _ohlc_csv_text(
            [
                ("2022.11.07.09:30:00", 1800.0, 1801.0, 1799.5, 1800.5, 100),
                ("2022.11.07.09:31:00", 1800.5, 1802.0, 1800.0, 1801.5, 120),
            ]
        )
    )
    df = load_contract(p, schema="ohlc")
    assert list(df.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["time"].dtype.kind == "M"
    assert df["open"].iloc[0] == 1800.0
    assert df["volume"].iloc[1] == 120.0


def test_drops_low_liquidity_days() -> None:
    full = _synth_day("2022-11-07", 480, "TEST22").drop(columns=["contract"])
    partial = _synth_day("2022-11-08", 200, "TEST22").drop(columns=["contract"])
    df = pd.concat([full, partial], ignore_index=True)
    out = drop_low_liquidity_days(df, min_fraction=0.90)
    kept_dates = out["time"].dt.date.unique().tolist()
    assert kept_dates == [pd.Timestamp("2022-11-07").date()]


def test_drop_low_liquidity_handles_empty() -> None:
    empty = pd.DataFrame({c: [] for c in ["time", "open", "high", "low", "close", "volume"]})
    empty["time"] = pd.to_datetime(empty["time"])
    out = drop_low_liquidity_days(empty)
    assert out.empty


def test_roll_picks_higher_volume_contract() -> None:
    # Day 1: A dominates volume. Day 2: B dominates. Expect roll at the crossover.
    a = pd.concat(
        [
            _synth_day("2022-11-07", 480, "A", vol=10.0),
            _synth_day("2022-11-08", 480, "A", vol=1.0),
        ],
        ignore_index=True,
    ).drop(columns=["contract"])
    b = pd.concat(
        [
            _synth_day("2022-11-07", 480, "B", vol=1.0),
            _synth_day("2022-11-08", 480, "B", vol=10.0),
        ],
        ignore_index=True,
    ).drop(columns=["contract"])

    out = pick_active_contract({"A": a, "B": b})
    by_date = out.groupby(out["time"].dt.date)["contract"].unique()
    assert list(by_date[pd.Timestamp("2022-11-07").date()]) == ["A"]
    assert list(by_date[pd.Timestamp("2022-11-08").date()]) == ["B"]


def test_pick_active_contract_output_is_monotonic_in_time() -> None:
    """No-lookahead invariant at the loader boundary: output timestamps are non-decreasing."""
    a = _synth_day("2022-11-07", 480, "A", vol=10.0).drop(columns=["contract"])
    b = _synth_day("2022-11-07", 480, "B", vol=1.0).drop(columns=["contract"])
    out = pick_active_contract({"A": a, "B": b})
    times = out["time"].tolist()
    assert times == sorted(times)


def test_pick_active_contract_handles_empty() -> None:
    out = pick_active_contract({})
    assert out.empty
    assert "contract" in out.columns


def test_market_spec_is_frozen() -> None:
    spec = MarketSpec(name="Gold", tick_size=0.10)
    with pytest.raises(FrozenInstanceError):
        spec.tick_size = 0.25  # type: ignore[misc]


def _day_bars(date_str: str, n_minutes: int, vol: float = 10.0) -> pd.DataFrame:
    """One day's bars starting at midnight so up to 1439 minutes stay within the date."""
    times = pd.date_range(pd.Timestamp(f"{date_str} 00:00:00"), periods=n_minutes, freq="1min")
    return pd.DataFrame(
        {"time": times, "open": 1000.0, "high": 1001.0, "low": 999.0, "close": 1000.0, "volume": vol}
    )


def test_roll_drops_loser_bars_each_side() -> None:
    """At the crossover the winner flips and the loser's bars are dropped on BOTH days."""
    a = pd.concat(
        [_synth_day("2022-11-07", 480, "A", vol=10.0), _synth_day("2022-11-08", 480, "A", vol=1.0)],
        ignore_index=True,
    ).drop(columns=["contract"])
    b = pd.concat(
        [_synth_day("2022-11-07", 480, "B", vol=1.0), _synth_day("2022-11-08", 480, "B", vol=10.0)],
        ignore_index=True,
    ).drop(columns=["contract"])

    out = pick_active_contract({"A": a, "B": b})
    per_date = out.groupby(out["time"].dt.date)["contract"]
    # Exactly one contract survives each day, and it's the volume winner.
    assert per_date.nunique().tolist() == [1, 1]
    assert out[out["time"].dt.date == pd.Timestamp("2022-11-07").date()]["contract"].eq("A").all()
    assert out[out["time"].dt.date == pd.Timestamp("2022-11-08").date()]["contract"].eq("B").all()
    # Loser bars are gone, not just hidden: 480 bars/day survive, not 960.
    assert len(out) == 960


def test_liquidity_relative_boundary() -> None:
    """At fraction=0.90 vs the trailing typical session: keep 91%, drop 89%."""
    full = [_day_bars(f"2022-11-{d:02d}", 1000) for d in range(1, 22)]  # 21 full sessions
    day_91 = _day_bars("2022-11-22", 910)
    day_89 = _day_bars("2022-11-23", 890)
    df = pd.concat([*full, day_91, day_89], ignore_index=True)

    kept = set(drop_low_liquidity_days(df, min_fraction=0.90)["time"].dt.date)
    assert pd.Timestamp("2022-11-22").date() in kept       # 91% of typical -> kept
    assert pd.Timestamp("2022-11-23").date() not in kept   # 89% of typical -> dropped


def test_liquidity_robust_to_single_long_day() -> None:
    """A single anomalously long session must not drop normal full days (quantile, not max)."""
    normal = [_day_bars(f"2022-11-{d:02d}", 600) for d in range(1, 22)]
    anomaly = _day_bars("2022-11-22", 1400)   # one ~23h session inflates max but not the p95
    after = _day_bars("2022-11-23", 600)
    df = pd.concat([*normal, anomaly, after], ignore_index=True)

    kept = set(drop_low_liquidity_days(df, min_fraction=0.90)["time"].dt.date)
    # Under a max-based rule (0.90*1400=1260) this 600-bar day would wrongly drop.
    assert pd.Timestamp("2022-11-23").date() in kept


def test_active_day_mask_drops_only_empty_valley() -> None:
    # Bimodal-ish: one near-empty day amid full sessions. p95 lands on the full session
    # (~100), so 0.90*100=90 keeps the full days and drops the 50-bar day.
    counts = pd.Series([100, 100, 100, 50, 100], index=pd.date_range("2022-01-01", periods=5))
    mask = active_day_mask(counts, 0.90)
    assert mask.tolist() == [True, True, True, False, True]


def test_active_day_mask_handles_empty() -> None:
    mask = active_day_mask(pd.Series(dtype=float))
    assert mask.empty
