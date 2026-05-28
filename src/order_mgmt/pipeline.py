"""Pipeline glue: roll-aware multi-contract loading in the team's indexed-by-time format.

Bridges `order_mgmt.loader.load_market` (returns time-as-column with contract tag) into
the indexed-by-time DataFrame shape the team's `compute_all_ranges` / `_compute_stats`
expect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from order_mgmt.loader import MarketSpec, load_market

_INDEXED_COLUMNS = ["open", "high", "low", "close", "volume", "contract"]


def load_market_indexed(
    market_dir: Path | str,
    *,
    min_fraction: float = 0.90,
) -> pd.DataFrame:
    """Roll-aware multi-contract load → DataFrame indexed by 1-minute timestamps.

    Index: DatetimeIndex named 'time'.
    Columns: open, high, low, close, volume, contract.
    """
    market_dir = Path(market_dir)
    spec = MarketSpec(name=market_dir.name, tick_size=0.0)
    df = load_market(market_dir, spec, min_fraction=min_fraction)
    if df.empty:
        empty_idx = pd.DatetimeIndex([], name="time")
        return pd.DataFrame({c: [] for c in _INDEXED_COLUMNS}, index=empty_idx)
    return df.set_index("time").sort_index()
