"""Data loader: 1-min OHLC futures bars, contract-roll-aware, liquidity-filtered.

Generic across markets via `MarketSpec`. Downstream modules consume the rolled,
filtered DataFrame and never re-read raw CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Schema = Literal["ohlc", "aiagent"]

EXPECTED_MIN_PER_DAY = 480
MIN_FRACTION_FOR_FULL_DAY = 0.90

OHLC_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class MarketSpec:
    """Per-market configuration. Pass this in; never hardcode constants downstream."""

    name: str
    tick_size: float


def load_contract(path: Path | str, schema: Schema = "ohlc") -> pd.DataFrame:
    """Load a single contract CSV. Returns DataFrame typed and time-parsed.

    `schema="ohlc"` parses `YYYY.MM.DD.HH:MM:SS,o,h,l,c,v`.
    `schema="aiagent"` parses `day,hour,minute,price,extra` (different file family, kept separate).
    """
    path = Path(path)
    if schema == "ohlc":
        df = pd.read_csv(
            path,
            header=None,
            names=OHLC_COLUMNS,
            dtype={
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            },
        )
        df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d.%H:%M:%S")
        return df
    if schema == "aiagent":
        return pd.read_csv(
            path,
            header=None,
            names=["day", "hour", "minute", "price", "extra"],
        )
    raise ValueError(f"Unknown schema: {schema}")


def drop_low_liquidity_days(
    df: pd.DataFrame, min_fraction: float = MIN_FRACTION_FOR_FULL_DAY
) -> pd.DataFrame:
    """Drop days with fewer than `min_fraction * EXPECTED_MIN_PER_DAY` bars."""
    if df.empty:
        return df
    by_date = df.assign(_date=df["time"].dt.date)
    counts = by_date.groupby("_date").size()
    keep = counts[counts >= min_fraction * EXPECTED_MIN_PER_DAY].index
    return by_date[by_date["_date"].isin(keep)].drop(columns=["_date"]).reset_index(drop=True)


def pick_active_contract(per_contract: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-contract bars; on each date, keep only the highest-volume contract.

    Roll happens implicitly at the daily-volume crossover, which is what the spec describes
    (Figure 1: ESH20 → ESM20 around 2020-03-17).
    """
    if not per_contract:
        return pd.DataFrame(columns=[*OHLC_COLUMNS, "contract"])

    frames = []
    for symbol, df in per_contract.items():
        if df.empty:
            continue
        tagged = df.copy()
        tagged["contract"] = symbol
        frames.append(tagged)
    if not frames:
        return pd.DataFrame(columns=[*OHLC_COLUMNS, "contract"])

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["_date"] = all_rows["time"].dt.date
    daily_vol = all_rows.groupby(["_date", "contract"])["volume"].sum().unstack(fill_value=0.0)
    winners = daily_vol.idxmax(axis=1)
    all_rows["_winner"] = all_rows["_date"].map(winners)
    kept = all_rows[all_rows["contract"] == all_rows["_winner"]].drop(columns=["_date", "_winner"])
    return kept.sort_values("time").reset_index(drop=True)


def load_market(
    market_dir: Path | str,
    spec: MarketSpec,
    *,
    min_fraction: float = MIN_FRACTION_FOR_FULL_DAY,
) -> pd.DataFrame:
    """Load all OHLC contracts in `market_dir`, roll between them, drop low-liquidity days.

    Returns columns `[time, open, high, low, close, volume, contract]`, sorted by time.
    `AIAgent_*.csv` files are skipped — they have a different schema (use `load_contract`
    with `schema="aiagent"` if you need them).
    """
    market_dir = Path(market_dir)
    csv_paths = [p for p in sorted(market_dir.glob("*.csv")) if not p.stem.startswith("AIAgent_")]
    per_contract = {p.stem: load_contract(p, schema="ohlc") for p in csv_paths}
    rolled = pick_active_contract(per_contract)
    return drop_low_liquidity_days(rolled, min_fraction=min_fraction)
