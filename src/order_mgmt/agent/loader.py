"""Load and decode `AIAgent_*.csv` decision series into tidy trade orders.

The AIAgent files have NO header. Schema (assigned by us, verified against the data):

    day, hour, minute, price, extra

  - `day`   : Excel serial day (base 1899-12-30); e.g. 45293 -> 2024-01-02.
  - `hour`  : 0..23
  - `minute`: on a 5-minute grid {0,5,...,55}
  - `price` : the agent's decision price at that 5-minute mark
  - `extra` : the agent's RUNNING SIGNED POSITION (inventory), e.g. -7..+8 for Gold.
              It is NOT volume and NOT a binary flag — verified from the data
              (Gold takes integer values -7..+8 across the file).

Trade direction (the only place direction is inferred):
    The parent orders are the rows where the position CHANGES.
        dpos = position.diff()
        dpos > 0 -> buy  dpos        units
        dpos < 0 -> sell |dpos|      units
        dpos == 0 -> no order
    `dpos` is known at the decision time (it is the agent's own action), so this is
    causal — NO lookahead. Inferring direction from price[i+1]-price[i] WOULD be
    lookahead and is deliberately not done.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from order_mgmt.loader import load_contract

# Excel/Lotus serial-day epoch. Pandas, Excel and this dataset all count days from here.
EXCEL_EPOCH = pd.Timestamp("1899-12-30")


@dataclass(frozen=True)
class AgentSeries:
    """A decoded AIAgent decision series.

    `df` is indexed by a DatetimeIndex named 'time' and has columns:
        price    : decision price
        position : running signed inventory (the raw `extra` column)
        dpos     : position.diff() (NaN on the first row)
        side     : "buy" | "sell" | None   (None where dpos is 0 or NaN)
        qty      : |dpos| as int (0 where no trade)
    """

    market: str
    df: pd.DataFrame


def decode_agent_timestamps(raw: pd.DataFrame) -> pd.DatetimeIndex:
    """Decode (day, hour, minute) -> DatetimeIndex via the Excel serial-day epoch.

    Vectorized; no timezone. `45293, 0, 5` -> 2024-01-02 00:05:00.
    """
    ts = (
        EXCEL_EPOCH
        + pd.to_timedelta(raw["day"].to_numpy(), unit="D")
        + pd.to_timedelta(raw["hour"].to_numpy(), unit="h")
        + pd.to_timedelta(raw["minute"].to_numpy(), unit="m")
    )
    return pd.DatetimeIndex(ts, name="time")


def load_agent_series(path: Path | str, market: str) -> AgentSeries:
    """Load an `AIAgent_*.csv` and decode it into a tidy `AgentSeries`.

    Reuses `order_mgmt.loader.load_contract(schema="aiagent")` for the raw parse.
    """
    raw = load_contract(path, schema="aiagent")
    idx = decode_agent_timestamps(raw)

    position = raw["extra"].astype(float)  # the `extra` column IS the running position
    dpos = position.diff()

    # Direction from the agent's own position change (causal — see module docstring).
    side = pd.Series(np.where(dpos > 0, "buy", np.where(dpos < 0, "sell", None)), index=raw.index)
    qty = dpos.abs().fillna(0).astype(int)

    df = pd.DataFrame(
        {
            "price": raw["price"].astype(float).to_numpy(),
            "position": position.to_numpy(),
            "dpos": dpos.to_numpy(),
            "side": side.to_numpy(),
            "qty": qty.to_numpy(),
        },
        index=idx,
    ).sort_index()
    return AgentSeries(market=market, df=df)


def trade_decisions(series: AgentSeries) -> pd.DataFrame:
    """Rows that are actual parent orders (position changed -> side is buy/sell)."""
    df = series.df
    return df[df["side"].notna()].copy()


# --- Market / file resolution ------------------------------------------------


def find_market_dir(data_root: Path | str, substring: str) -> Path | None:
    """Find a market directory under `data_root` by case-insensitive substring.

    Mirrors `scripts/run_v1.py:_find_market_dir` so market naming is consistent.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        return None
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and substring.lower() in p.name.lower():
            return p
    return None


def find_agent_csv(market_dir: Path | str) -> Path | None:
    """Return the AIAgent_*.csv inside a market directory, if any."""
    market_dir = Path(market_dir)
    matches = sorted(market_dir.glob("AIAgent_*.csv"))
    return matches[0] if matches else None


def load_market_for_agent(market_dir: Path | str) -> tuple[pd.DataFrame, float, list, str]:
    """Load the OHLC side for a market: rolled, liquidity-filtered, with tick + proper days.

    Returns (df_ohlcv, tick, proper_days, first_contract_stem). `df_ohlcv` is indexed by
    time with columns [open, high, low, close, volume] (the 'contract' column dropped).
    Mirrors the wiring in `scripts/run_v1.py` (load_market_indexed -> _compute_stats ->
    resolve_tick) so tick handling is identical to the rest of the pipeline.
    """
    # Imported here to keep loader import-light and avoid a matplotlib import at module load.
    from order_mgmt.pipeline import load_market_indexed
    from order_mgmt.ticks import resolve_tick
    from plot_volume import _compute_stats

    market_dir = Path(market_dir)
    df_indexed = load_market_indexed(market_dir)
    if df_indexed.empty:
        return df_indexed, 0.0, [], ""

    first_stem = df_indexed["contract"].unique().tolist()[0]
    df_ohlcv = df_indexed[["open", "high", "low", "close", "volume"]]
    _, inferred_tick, proper_days, _n_green, _n_total = _compute_stats(df_ohlcv)
    tick = resolve_tick(first_stem, inferred_tick)
    return df_ohlcv, tick, proper_days, first_stem
