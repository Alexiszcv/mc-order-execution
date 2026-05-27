"""Naive execution baselines: TWAP and VWAP, used to benchmark the conditional strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    """
    realized: list[float] = []
    benchmark: list[float] = []
    slippage: list[float] = []

    dt_tau = pd.Timedelta(minutes=tau)
    opens_series = df_1min["open"]

    for t in t_list:
        window = df_1min.loc[(df_1min.index >= t) & (df_1min.index < t + dt_tau)]
        if window.empty:
            continue
        typ = (window["high"] + window["low"] + window["close"]) / 3.0
        v = window["volume"]
        if v.sum() == 0:
            vwap = float(typ.mean())
        else:
            vwap = float((typ * v).sum() / v.sum())
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
