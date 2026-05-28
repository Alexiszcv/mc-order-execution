"""Backtest the regime-conditioned limit-order strategy on historical OHLC data.

Two variants:
  - `run_backtest` (v1): uses the ePDFs and quantile thresholds built from the FULL
    history at every decision. Permissive — quantifies the maximum achievable edge
    under perfect information about marginal distributions.
  - `run_backtest_rolling` (v2): streaming, no-lookahead. At each decision j, the ePDFs
    and quantile thresholds use only windows j_start..j-1. Mirrors the build_epdf
    streaming pattern; same O(n log n) complexity.
"""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# Team modules (flat src/ layout — pythonpath = ["src"] is set in pyproject.toml).
from epdf import build_epdf
from ranges import compute_all_ranges
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


def run_backtest_rolling(
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
    """v2 backtest: strict no-lookahead.

    Streams windows in order. At each j ≥ j_start:
      1. Classify regime from j-1 EWMA stats using quantile thresholds derived
         from a sorted prefix of windows STRICTLY before j-1.
      2. Look up the per-regime ePDF, built from earlier observations only.
      3. Pick ℓ*, simulate fill against realized ell_u[j] / ell_d[j].
      4. After the decision, fold j-1's observation into the sorted lists AND
         increment counts at (regime, ell_u[j]) for future cycles.

    Same O(n log n) cost as `epdf.build_epdf`.
    """
    t_list, _ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(
        df_1min, tau, tick, proper_days
    )
    if len(t_list) <= j_start:
        return BacktestResult(side, 0, 0, 0.0, [], [], [], 0.0, 0.0, 0.0)

    ewma_range, ewma_vol = compute_ewma_series(t_list, _ell_r, vol_list, half_life)
    ev = np.asarray(ewma_vol, dtype=float)
    er = np.asarray(ewma_range, dtype=float)
    dx_arr = np.asarray(dx_list, dtype=float)

    # Pre-populate sorted lists with the prefix strictly before j_start - 1
    pre = max(0, j_start - 1)
    sv = sorted(float(v) for v in ev[:pre] if not np.isnan(v))
    sr = sorted(float(v) for v in er[:pre] if not np.isnan(v))
    sd = sorted(float(v) for v in dx_arr[:pre] if not np.isnan(v))

    # Per-regime ePDF for the side we trade (R_U for sells, R_D for buys)
    counts_decision: dict[tuple[int, int, int], Counter] = {}

    opens_series = df_1min["open"]
    closes_series = df_1min["close"]
    dt_last = pd.Timedelta(minutes=tau - 1)

    realized: list[float] = []
    benchmark: list[float] = []
    slippage: list[float] = []
    n_filled = 0

    for j in range(max(j_start, 1), len(t_list)):
        v_prev = ev[j - 1]
        r_prev = er[j - 1]
        d_prev = dx_arr[j - 1]
        cell: tuple[int, int, int] | None = None
        if not (np.isnan(v_prev) or np.isnan(r_prev) or np.isnan(d_prev)):
            m = _state(float(v_prev), sv, M)
            n = _state(float(r_prev), sr, N)
            k = _state(float(d_prev), sd, K)
            if m is not None and n is not None and k is not None:
                cell = (m, n, k)

        if cell is not None:
            epdf = counts_decision.get(cell, Counter())
            if epdf:
                ell_star = pick_ell_star(epdf, fill_rate_target)
                t = t_list[j]
                try:
                    open_j = float(opens_series.loc[t])
                    close_j = float(closes_series.loc[t + dt_last])
                except KeyError:
                    open_j = close_j = None

                if open_j is not None:
                    if side == "sell":
                        if ell_u[j] >= ell_star:
                            price = open_j + ell_star * tick
                            n_filled += 1
                        else:
                            price = close_j
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

        # Fold j-1's EWMA values into the sorted lists; fold window j's outcome
        # into the counter at (j-1's regime) so it's available next cycle.
        if not np.isnan(v_prev):
            bisect.insort(sv, float(v_prev))
        if not np.isnan(r_prev):
            bisect.insort(sr, float(r_prev))
        if not np.isnan(d_prev):
            bisect.insort(sd, float(d_prev))

        if cell is not None:
            if cell not in counts_decision:
                counts_decision[cell] = Counter()
            ell_obs = ell_u[j] if side == "sell" else ell_d[j]
            counts_decision[cell][ell_obs] += 1

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
