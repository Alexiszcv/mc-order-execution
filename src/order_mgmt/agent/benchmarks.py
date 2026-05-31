"""Benchmark the regime-conditioned execution against dumb / no-skill baselines.

To justify the method we must show it beats strategies that use less (or no) skill. On
the SAME agent orders we compare:

  - market   : all-in at the window open, no limit (the do-nothing-clever floor).
  - random   : post a limit at a RANDOM offset (seeded). Control for "is picking ℓ*
               meaningfully better than guessing an offset?"
  - global   : limit at ℓ* from the GLOBAL pooled ePDF (all windows, regime IGNORED).
               THE ablation — if `regime` ≈ `global`, regime conditioning adds nothing.
  - regime   : limit at ℓ* from the regime-conditioned ePDF (our core method).
  - dp       : the dynamic backward-induction policy (our complex strategy).

All share the regime fit, the window bars, and the no-lookahead train/eval split, so the
only thing that varies is the decision rule. Returns {name: {"shortfall": ndarray}}.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from order_mgmt.agent.dynamic import dp_schedule, fit_horizon_tables, simulate_dynamic
from order_mgmt.agent.loader import AgentSeries, trade_decisions
from order_mgmt.agent.metrics import _fit_regime_model, _window_bars
from order_mgmt.agent.slicing import fill_capped, fill_single
from order_mgmt.backtest import Side, _state
from order_mgmt.strategy import pick_ell_star
from ranges import compute_all_ranges
from regime import compute_ewma_series

NAMES = ["market", "random", "global", "regime", "dp"]


def _short(side: Side, price: float, arrival: float, tick: float) -> float:
    return (price - arrival) / tick if side == "sell" else (arrival - price) / tick


def evaluate_agent_benchmarks(
    agent: AgentSeries,
    df_ohlcv: pd.DataFrame,
    tick: float,
    proper_days: list,
    *,
    tau: int,
    half_life: int,
    M: int,
    N: int,
    K: int,
    fill_rate_target: float,
    j_start: int,
    seed: int = 0,
    train_end: pd.Timestamp | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    empty = {nm: {"shortfall": np.array([])} for nm in NAMES}
    trades = trade_decisions(agent)

    t_list, ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(df_ohlcv, tau, tick, proper_days)
    if not t_list or trades.empty:
        return empty
    ewma_range, ewma_vol = compute_ewma_series(t_list, ell_r, vol_list, half_life)
    ev = np.asarray(ewma_vol, dtype=float)
    er = np.asarray(ewma_range, dtype=float)
    dx = np.asarray(dx_list, dtype=float)

    if train_end is None:
        train_end = trades.index.min()
    t_ns_arr = np.asarray([pd.Timestamp(t).value for t in t_list], dtype=np.int64)
    train_count = int(np.searchsorted(t_ns_arr, pd.Timestamp(train_end).value, side="left"))
    counts_RU, counts_RD, sv, sr, sd = _fit_regime_model(ell_u, ell_d, ev, er, dx, train_count, M, N, K, j_start)

    # GLOBAL pooled ePDF (regime ignored) over the same training windows.
    g_RU: Counter = Counter()
    g_RD: Counter = Counter()
    for j in range(max(j_start, 1), train_count):
        g_RU[ell_u[j]] += 1
        g_RD[ell_d[j]] += 1
    global_ell = {"sell": pick_ell_star(g_RU, fill_rate_target), "buy": pick_ell_star(g_RD, fill_rate_target)}
    rand_cap = max(2 * max(global_ell.values()), 4)

    if not df_ohlcv.index.is_monotonic_increasing:
        df_ohlcv = df_ohlcv.sort_index()
    idx_ns = df_ohlcv.index.as_unit("ns").asi8
    opens, highs, lows, closes = (df_ohlcv[c].to_numpy() for c in ("open", "high", "low", "close"))
    tau_ns = int(pd.Timedelta(minutes=tau).value)

    tables = fit_horizon_tables(
        t_ns_arr, ev, er, dx, sv, sr, sd, M, N, K, j_start, train_count,
        idx_ns, opens, highs, lows, closes, tau, tau_ns, tick,
    )
    dp_cache: dict[tuple, dict] = {}
    out: dict[str, list[float]] = {nm: [] for nm in NAMES}

    for t, row in trades.iterrows():
        side: Side = row["side"]
        t_ns = int(pd.Timestamp(t).value)
        w = int(np.searchsorted(t_ns_arr, t_ns - tau_ns, side="right")) - 1
        if w < 1:
            continue
        v_prev, r_prev, d_prev = ev[w], er[w], dx[w]
        if np.isnan(v_prev) or np.isnan(r_prev) or np.isnan(d_prev):
            continue
        m = _state(float(v_prev), sv, M)
        n = _state(float(r_prev), sr, N)
        k = _state(float(d_prev), sd, K)
        if m is None or n is None or k is None:
            continue
        cell = (m, n, k)
        epdf = (counts_RU if side == "sell" else counts_RD).get(cell, Counter())
        if not epdf or cell not in tables:
            continue
        regime_ell = pick_ell_star(epdf, fill_rate_target)

        bars = _window_bars(idx_ns, opens, highs, lows, closes, t_ns, tau_ns)
        if bars is None:
            continue
        o, h, low_arr, c = bars
        bench = float(o[0])  # basis-immune benchmark: the OHLC window open, not the agent price

        # market: ℓ=0 ⇒ fill at the open; the other static strategies vary only the offset.
        out["market"].append(_short(side, fill_single(side, 0, o, h, low_arr, c, tick)[0], bench, tick))
        rand_ell = int(rng.integers(0, rand_cap + 1))
        out["random"].append(_short(side, fill_single(side, rand_ell, o, h, low_arr, c, tick)[0], bench, tick))
        out["global"].append(_short(side, fill_single(side, global_ell[side], o, h, low_arr, c, tick)[0], bench, tick))
        out["regime"].append(_short(side, fill_single(side, regime_ell, o, h, low_arr, c, tick)[0], bench, tick))

        key = (cell, side)
        if key not in dp_cache:
            dp_cache[key] = dp_schedule(tables[cell], side, tau)
        out["dp"].append(simulate_dynamic(side, dp_cache[key], o, h, low_arr, c, bench, tick, tau)[0])

    return {nm: {"shortfall": np.array(out[nm], dtype=float)} for nm in NAMES}


def evaluate_tail_strategies(
    agent: AgentSeries,
    df_ohlcv: pd.DataFrame,
    tick: float,
    proper_days: list,
    *,
    tau: int,
    half_life: int,
    M: int,
    N: int,
    K: int,
    fill_rate_target: float,
    j_start: int,
    caps: tuple[int, ...] = (4, 6, 8, 10, 12, 16),
    train_end: pd.Timestamp | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Tail-reduction comparison: market / regime / dp / capped@each cap (one pass).

    The capped strategy posts the regime ℓ* but stops out at `cap` ticks of adverse move —
    keeping the limit upside while truncating the chase tail. Sweeping `cap` traces the
    mean/tail frontier and reveals the cap that maximises the (risk-adjusted) mean.
    """
    cap_names = [f"cap{c}" for c in caps]
    names = ["market", "regime", "dp", *cap_names]
    empty = {nm: {"shortfall": np.array([])} for nm in names}
    trades = trade_decisions(agent)

    t_list, ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(df_ohlcv, tau, tick, proper_days)
    if not t_list or trades.empty:
        return empty
    ewma_range, ewma_vol = compute_ewma_series(t_list, ell_r, vol_list, half_life)
    ev = np.asarray(ewma_vol, dtype=float)
    er = np.asarray(ewma_range, dtype=float)
    dx = np.asarray(dx_list, dtype=float)

    if train_end is None:
        train_end = trades.index.min()
    t_ns_arr = np.asarray([pd.Timestamp(t).value for t in t_list], dtype=np.int64)
    train_count = int(np.searchsorted(t_ns_arr, pd.Timestamp(train_end).value, side="left"))
    counts_RU, counts_RD, sv, sr, sd = _fit_regime_model(ell_u, ell_d, ev, er, dx, train_count, M, N, K, j_start)

    if not df_ohlcv.index.is_monotonic_increasing:
        df_ohlcv = df_ohlcv.sort_index()
    idx_ns = df_ohlcv.index.as_unit("ns").asi8
    opens, highs, lows, closes = (df_ohlcv[col].to_numpy() for col in ("open", "high", "low", "close"))
    tau_ns = int(pd.Timedelta(minutes=tau).value)

    tables = fit_horizon_tables(
        t_ns_arr, ev, er, dx, sv, sr, sd, M, N, K, j_start, train_count,
        idx_ns, opens, highs, lows, closes, tau, tau_ns, tick,
    )
    dp_cache: dict[tuple, dict] = {}
    out: dict[str, list[float]] = {nm: [] for nm in names}

    for t, row in trades.iterrows():
        side: Side = row["side"]
        t_ns = int(pd.Timestamp(t).value)
        w = int(np.searchsorted(t_ns_arr, t_ns - tau_ns, side="right")) - 1
        if w < 1:
            continue
        v_prev, r_prev, d_prev = ev[w], er[w], dx[w]
        if np.isnan(v_prev) or np.isnan(r_prev) or np.isnan(d_prev):
            continue
        m = _state(float(v_prev), sv, M)
        n = _state(float(r_prev), sr, N)
        k = _state(float(d_prev), sd, K)
        if m is None or n is None or k is None:
            continue
        cell = (m, n, k)
        epdf = (counts_RU if side == "sell" else counts_RD).get(cell, Counter())
        if not epdf or cell not in tables:
            continue
        regime_ell = pick_ell_star(epdf, fill_rate_target)
        bars = _window_bars(idx_ns, opens, highs, lows, closes, t_ns, tau_ns)
        if bars is None:
            continue
        o, h, low_arr, c = bars
        bench = float(o[0])  # basis-immune benchmark: the OHLC window open, not the agent price

        out["market"].append(_short(side, fill_single(side, 0, o, h, low_arr, c, tick)[0], bench, tick))
        out["regime"].append(_short(side, fill_single(side, regime_ell, o, h, low_arr, c, tick)[0], bench, tick))
        key = (cell, side)
        if key not in dp_cache:
            dp_cache[key] = dp_schedule(tables[cell], side, tau)
        out["dp"].append(simulate_dynamic(side, dp_cache[key], o, h, low_arr, c, bench, tick, tau)[0])
        for cap, nm in zip(caps, cap_names, strict=True):
            out[nm].append(_short(side, fill_capped(side, regime_ell, o, h, low_arr, c, tick, cap)[0], bench, tick))

    return {nm: {"shortfall": np.array(out[nm], dtype=float)} for nm in names}
