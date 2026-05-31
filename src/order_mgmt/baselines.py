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
    Slippage = (vwap − open) / tick for sell, (open − vwap) / tick for buy.

    Each window [t, t+τ) is located by `searchsorted` on the int64-ns index (O(log n)
    per window, O(n log n) overall) instead of a per-window boolean-mask `.loc`
    (O(n²)). Behaviour is identical to the mask version — see test_backtest.py.
    """
    if not t_list:
        return BaselineResult("VWAP", side, 0, [], [], [])

    dt_tau_ns = int(pd.Timedelta(minutes=tau).value)
    opens_series = df_1min["open"]

    # int64-ns is unit-correct on pandas 3.0+ (default datetime unit is µs, not ns).
    idx_ns = df_1min.index.as_unit("ns").asi8
    typ = ((df_1min["high"] + df_1min["low"] + df_1min["close"]) / 3.0).to_numpy()
    vol = df_1min["volume"].to_numpy()
    if not df_1min.index.is_monotonic_increasing:
        order = np.argsort(idx_ns, kind="stable")
        idx_ns, typ, vol = idx_ns[order], typ[order], vol[order]

    realized: list[float] = []
    benchmark: list[float] = []
    slippage: list[float] = []

    for t in t_list:
        t_ns = int(pd.Timestamp(t).value)
        lo = int(np.searchsorted(idx_ns, t_ns, side="left"))
        hi = int(np.searchsorted(idx_ns, t_ns + dt_tau_ns, side="left"))
        if hi <= lo:
            continue
        w_typ = typ[lo:hi]
        w_vol = vol[lo:hi]
        vsum = w_vol.sum()
        vwap = float(w_typ.mean()) if vsum == 0 else float((w_typ * w_vol).sum() / vsum)
        open_j = float(opens_series.loc[t])
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
