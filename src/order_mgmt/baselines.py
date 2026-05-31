"""Naive execution baselines: TWAP and VWAP, used to benchmark the conditional strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class BaselineResult:
    name: str
    side: Side
    n_windows: int
    realized_prices: list[float]
    benchmark_prices: list[float]
    slippage_ticks: list[float]


def twap_baseline(opens: list[float], side: Side) -> BaselineResult:
    """TWAP: market-execute at the open of each window. Slippage vs open is zero by definition."""
    return BaselineResult(
        name="TWAP",
        side=side,
        n_windows=len(opens),
        realized_prices=list(opens),
        benchmark_prices=list(opens),
        slippage_ticks=[0.0] * len(opens),
    )


def vwap_baseline(
    df_1min: pd.DataFrame,
    t_list: list,
    tau: int,
    tick: float,
    side: Side,
) -> BaselineResult:
    """VWAP per window using bar-typical-price weighted by bar-volume.

    Bar typical price = (high + low + close) / 3.
    Slippage = (vwap - open) / tick for sell, (open - vwap) / tick for buy.

    Each window [t, t+τ) is a contiguous slice of the (sorted) 1-min index, so
    window bounds come from `searchsorted` (O(log n)) rather than a full boolean
    mask per window (O(n)). Output is identical to the masked version.
    """
    realized: list[float] = []
    benchmark: list[float] = []
    slippage: list[float] = []

    dt_tau = pd.Timedelta(minutes=tau)
    if not df_1min.index.is_monotonic_increasing:
        df_1min = df_1min.sort_index()

    # Precompute typical price, volume and open as numpy arrays once.
    idx = df_1min.index.values  # sorted datetime64[ns]
    typ_all = ((df_1min["high"] + df_1min["low"] + df_1min["close"]) / 3.0).to_numpy()
    vol_all = df_1min["volume"].to_numpy()
    open_all = df_1min["open"].to_numpy()
    dt_tau_ns = np.timedelta64(dt_tau)

    for t in t_list:
        t64 = np.datetime64(t)
        lo = int(np.searchsorted(idx, t64, side="left"))
        hi = int(np.searchsorted(idx, t64 + dt_tau_ns, side="left"))
        if hi <= lo:
            continue
        typ = typ_all[lo:hi]
        v = vol_all[lo:hi]
        vsum = v.sum()
        vwap = float(typ.mean()) if vsum == 0 else float((typ * v).sum() / vsum)
        open_j = float(open_all[lo])  # idx[lo] == t (window start is an actual bar)
        realized.append(vwap)
        benchmark.append(open_j)
        if side == "sell":
            slippage.append((vwap - open_j) / tick)
        else:
            slippage.append((open_j - vwap) / tick)

    return BaselineResult(
        name="VWAP",
        side=side,
        n_windows=len(realized),
        realized_prices=realized,
        benchmark_prices=benchmark,
        slippage_ticks=slippage,
    )
