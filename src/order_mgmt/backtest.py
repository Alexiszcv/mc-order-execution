"""Backtest the regime-conditioned limit-order strategy on historical OHLC data.

v1 simplification (documented):
    The conditional ePDFs and the regime thresholds are computed once from the FULL
    history (via `epdf.build_epdf`) and reused at every decision point. This is a
    permissive baseline — a proper backtest would rebuild ePDFs/thresholds incrementally
    so the choice at time t uses only data strictly before t. The v1 result is therefore
    an upper bound on strategy edge; v2 should re-do this with rolling estimation.
"""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# Team modules (flat src/ layout — pythonpath = ["src"] is set in pyproject.toml).
from epdf import build_epdf, compute_all_ranges
from regime import compute_ewma_series

from order_mgmt.strategy import pick_ell_star

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class BacktestResult:
    side: Side
    n_decisions: int
    n_filled: int
    fill_rate: float
    realized_prices: list[float]
    benchmark_prices: list[float]
    slippage_ticks: list[float]  # signed; positive = strategy beats open-price benchmark
    avg_slippage_ticks: float
    median_slippage_ticks: float
    total_slippage_ticks: float


def _state(value: float, sorted_arr: list[float], n_states: int) -> int | None:
    """Replicates the regime-state assignment used in `epdf.build_epdf`."""
    L = len(sorted_arr)
    if L < n_states:
        return None
    rank = bisect.bisect_left(sorted_arr, value)
    return min(rank * n_states // L + 1, n_states)


def _final_sorted(arr: np.ndarray) -> list[float]:
    return sorted(float(v) for v in arr if not np.isnan(v))


def run_backtest(
    df_1min: pd.DataFrame,
    tau: int,
    tick: float,
    proper_days: list,
    side: Side,
    *,
    fill_rate_target: float,
    half_life: int,
    M: int,
    N: int,
    K: int,
    j_start: int,
) -> BacktestResult:
    t_list, _ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(
        df_1min, tau, tick, proper_days
    )
    if len(t_list) <= j_start:
        return BacktestResult(side, 0, 0, 0.0, [], [], [], 0.0, 0.0, 0.0)

    ewma_range, ewma_vol = compute_ewma_series(t_list, _ell_r, vol_list, half_life)

    counts_RU, counts_RD, _thr = build_epdf(
        t_list,
        ell_u,
        ell_d,
        list(ewma_vol),
        list(ewma_range),
        dx_list,
        M=M,
        N=N,
        K=K,
        j_start=j_start,
    )

    # Final thresholds (v1 lookahead-permitting simplification — see module docstring).
    vol_sorted = _final_sorted(np.asarray(ewma_vol, dtype=float))
    range_sorted = _final_sorted(np.asarray(ewma_range, dtype=float))
    dx_sorted = _final_sorted(np.asarray(dx_list, dtype=float))

    counts_lookup = counts_RU if side == "sell" else counts_RD

    opens_series = df_1min["open"]
    closes_series = df_1min["close"]
    dt_last = pd.Timedelta(minutes=tau - 1)

    realized: list[float] = []
    benchmark: list[float] = []
    slippage: list[float] = []
    n_filled = 0

    ewma_vol_arr = np.asarray(ewma_vol, dtype=float)
    ewma_range_arr = np.asarray(ewma_range, dtype=float)
    dx_arr = np.asarray(dx_list, dtype=float)

    for j in range(max(j_start, 1), len(t_list)):
        v_prev = ewma_vol_arr[j - 1]
        r_prev = ewma_range_arr[j - 1]
        d_prev = dx_arr[j - 1]
        if np.isnan(v_prev) or np.isnan(r_prev) or np.isnan(d_prev):
            continue
        m = _state(float(v_prev), vol_sorted, M)
        n = _state(float(r_prev), range_sorted, N)
        k = _state(float(d_prev), dx_sorted, K)
        if m is None or n is None or k is None:
            continue

        epdf = counts_lookup.get((m, n, k), Counter())
        if not epdf:
            continue
        ell_star = pick_ell_star(epdf, fill_rate_target)

        t = t_list[j]
        try:
            open_j = float(opens_series.loc[t])
            close_j = float(closes_series.loc[t + dt_last])
        except KeyError:
            continue

        if side == "sell":
            # limit set ell_star ticks above open; fills iff R_U ≥ ell_star
            if ell_u[j] >= ell_star:
                price = open_j + ell_star * tick
                n_filled += 1
            else:
                price = close_j  # chase at window close
            slip = (price - open_j) / tick
        else:
            if ell_d[j] >= ell_star:
                price = open_j - ell_star * tick
                n_filled += 1
            else:
                price = close_j
            slip = (open_j - price) / tick

        realized.append(price)
        benchmark.append(open_j)
        slippage.append(slip)

    n_dec = len(realized)
    fill_rate = n_filled / n_dec if n_dec else 0.0
    avg = float(np.mean(slippage)) if slippage else 0.0
    med = float(np.median(slippage)) if slippage else 0.0
    tot = float(np.sum(slippage)) if slippage else 0.0

    return BacktestResult(
        side=side,
        n_decisions=n_dec,
        n_filled=n_filled,
        fill_rate=fill_rate,
        realized_prices=realized,
        benchmark_prices=benchmark,
        slippage_ticks=slippage,
        avg_slippage_ticks=avg,
        median_slippage_ticks=med,
        total_slippage_ticks=tot,
    )
