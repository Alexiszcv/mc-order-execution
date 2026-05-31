"""Synthetic AIAgent generator — the zero-shot / new-data stress test.

Produces an `AgentSeries` of the SAME shape `loader.load_agent_series` returns, so
the metrics path is byte-identical: running `evaluate_agent_execution` on a synthetic
series exercises the exact same code as a real `AIAgent_*.csv`. That is the genericity
proof — if the pipeline is overfit to Gold's quirks it will break here.

Everything is driven by a seeded `np.random.default_rng(seed)` for reproducibility.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from order_mgmt.agent.loader import AgentSeries

Mode = Literal["sample_closes", "random_walk"]


def synth_agent_series(
    df_ohlcv: pd.DataFrame,
    *,
    seed: int,
    n_decisions: int = 500,
    mode: Mode = "sample_closes",
    max_abs_position: int = 8,
    start_fraction: float = 0.5,
    market: str = "synthetic",
) -> AgentSeries:
    """Generate a synthetic agent decision series aligned to `df_ohlcv`'s coverage.

    - Decisions sit on a 5-minute grid inside the OHLC date span, confined to the LAST
      ``1 - start_fraction`` of the coverage (default: the second half) so there is
      prior OHLC history for the no-lookahead regime fit — mirroring the real AIAgent
      files, which trade in a recent window with years of history behind them. The
      window ends ≥ 1h before the last bar so executions have forward bars to fill.
    - `position` is a bounded integer random walk in [-max_abs_position, max_abs_position];
      trades are where it changes (same convention as the real data).
    - `mode="sample_closes"`: price = the OHLC close at-or-before each decision (a
      realistic arrival price). `mode="random_walk"`: a seeded Gaussian price walk.
    """
    rng = np.random.default_rng(seed)
    if df_ohlcv.empty:
        empty = pd.DataFrame(
            columns=["price", "position", "dpos", "side", "qty"],
            index=pd.DatetimeIndex([], name="time"),
        )
        return AgentSeries(market=market, df=empty)

    idx = df_ohlcv.index
    if not idx.is_monotonic_increasing:
        df_ohlcv = df_ohlcv.sort_index()
        idx = df_ohlcv.index

    # 5-min grid over the coverage, restricted to real trading dates, leaving a tail
    # margin so every decision has a forward window of bars.
    trading_dates = set(idx.normalize().unique())
    span = idx.max() - idx.min()
    lo_bound = (idx.min() + span * start_fraction).ceil("5min")
    grid = pd.date_range(lo_bound, idx.max() - pd.Timedelta(hours=1), freq="5min")
    grid = grid[[ts.normalize() in trading_dates for ts in grid]]
    if len(grid) == 0:
        empty = pd.DataFrame(
            columns=["price", "position", "dpos", "side", "qty"],
            index=pd.DatetimeIndex([], name="time"),
        )
        return AgentSeries(market=market, df=empty)

    n = min(n_decisions, len(grid))
    chosen_pos = np.sort(rng.choice(len(grid), size=n, replace=False))
    times = pd.DatetimeIndex(grid[chosen_pos], name="time")

    # Bounded integer position random walk.
    steps = rng.choice([-1, 0, 1], size=n)
    position = np.clip(np.cumsum(steps), -max_abs_position, max_abs_position).astype(float)
    dpos = np.concatenate([[np.nan], np.diff(position)])

    if mode == "sample_closes":
        closes = df_ohlcv["close"].to_numpy()
        idx_ns = idx.as_unit("ns").asi8
        t_ns = np.array([t.value for t in times], dtype=np.int64)
        # bar at-or-before each decision time (causal arrival price)
        loc = np.searchsorted(idx_ns, t_ns, side="right") - 1
        loc = np.clip(loc, 0, len(closes) - 1)
        price = closes[loc]
    else:  # random_walk
        start = float(df_ohlcv["close"].iloc[0])
        scale = float(np.nanstd(np.diff(df_ohlcv["close"].to_numpy()))) or 1.0
        price = start + np.cumsum(rng.normal(0.0, scale, size=n))

    side = np.where(dpos > 0, "buy", np.where(dpos < 0, "sell", None))
    qty = np.where(np.isnan(dpos), 0, np.abs(dpos)).astype(int)

    df = pd.DataFrame(
        {"price": price.astype(float), "position": position, "dpos": dpos, "side": side, "qty": qty},
        index=times,
    )
    return AgentSeries(market=market, df=df)
