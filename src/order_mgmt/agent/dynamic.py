"""Dynamic (sequential) execution: re-decide each minute as the deadline approaches.

A parent order ("buy 5 in the next τ min") is an optimal-stopping problem: at each
minute with r minutes remaining, post a limit ℓ ticks from the *current* price; if it
fills, done; else carry to the next minute with r−1 left; at the deadline, market.

Two policies, both driven by per-regime multi-horizon tables fit on training windows
(no lookahead) and then simulated forward on the real 1-min bars:

  - **adaptive** : ℓ_r = the largest offset still expected to fill within the remaining r
    minutes at the target rate (uses the r-minute survival q(r,ℓ|cell)). Naturally shrinks
    as r → 1 (less time ⇒ post tighter).
  - **dp**       : backward-induct the value function with the 1-minute fill prob q(1,ℓ|cell)
    and a per-cell wait penalty w (expected 1-min adverse excursion):
        V(0)=0;  V(r)=max_ℓ [ q(1,ℓ)·ℓ + (1−q(1,ℓ))·(V(r−1) − w) ];  ℓ*_r = argmax.
    This is the principled "best execution" under the OHLC-only model. The wait penalty w
    is the one explicit modelling choice — documented, isolated, easy to swap.

Sign convention matches the rest of the package: shortfall ticks, +ve = beat arrival.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from order_mgmt.agent.loader import AgentSeries, trade_decisions
from order_mgmt.agent.metrics import _fit_regime_model, _window_bars
from order_mgmt.agent.slicing import fill_blend, fill_cutoff, fill_single
from order_mgmt.backtest import Side, _state
from order_mgmt.strategy import pick_ell_star
from ranges import compute_all_ranges
from regime import compute_ewma_series

Cell = tuple[int, int, int]


def _first_r_halfranges(o, h, low, c, r: int, tick: float) -> tuple[int, int]:
    """(R_U, R_D) in ticks over the first r bars of a window (favorable side depends on side)."""
    o0 = float(o[0])
    hi = float(h[:r].max())
    lo = float(low[:r].min())
    return max(round((hi - o0) / tick), 0), max(round((o0 - lo) / tick), 0)


def fit_horizon_tables(
    t_ns_arr, ev, er, dx, sv, sr, sd, M, N, K, j_start, train_count,
    idx_ns, opens, highs, lows, closes, tau_max, tau_max_ns, tick,
) -> dict[Cell, dict[str, dict[int, list[int]]]]:
    """Per-cell favorable/adverse half-range samples for horizons r=1..τ_max (training only).

    For each training window (cell read from the prior window, same as `_fit_regime_model`),
    record R_U and R_D over its first r minutes, r=1..τ_max. # INVARIANT: training windows
    only (index < train_count ⇒ time < train_end), so no lookahead.
    """
    tables: dict[Cell, dict[str, dict[int, list[int]]]] = {}
    for j in range(max(j_start, 1), train_count):
        v_prev, r_prev, d_prev = ev[j - 1], er[j - 1], dx[j - 1]
        if np.isnan(v_prev) or np.isnan(r_prev) or np.isnan(d_prev):
            continue
        m = _state(float(v_prev), sv, M)
        n = _state(float(r_prev), sr, N)
        k = _state(float(d_prev), sd, K)
        if m is None or n is None or k is None:
            continue
        bars = _window_bars(idx_ns, opens, highs, lows, closes, int(t_ns_arr[j]), tau_max_ns)
        if bars is None:
            continue
        o, h, low, c = bars
        cell = (m, n, k)
        cd = tables.setdefault(cell, {"RU": {}, "RD": {}})
        for rr in range(1, min(tau_max, len(o)) + 1):
            ru, rd = _first_r_halfranges(o, h, low, c, rr, tick)
            cd["RU"].setdefault(rr, []).append(ru)
            cd["RD"].setdefault(rr, []).append(rd)
    return tables


def _survival(values: list[int], ell: int) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values) >= ell))


def adaptive_schedule(cd: dict, side: Side, target: float, tau_max: int) -> dict[int, int]:
    """ℓ_r = largest offset with r-minute fill prob ≥ target (shrinks as r → 1)."""
    fav = cd["RD"] if side == "buy" else cd["RU"]
    sched: dict[int, int] = {}
    for r in range(1, tau_max + 1):
        vals = fav.get(r, [])
        best = 0
        if vals:
            for ell in range(0, int(max(vals)) + 1):
                if _survival(vals, ell) >= target:
                    best = ell
                else:
                    break  # survival is non-increasing in ell
        sched[r] = best
    return sched


def dp_schedule(cd: dict, side: Side, tau_max: int) -> dict[int, int]:
    """Backward-induct the optimal per-minute offset (see module docstring)."""
    fav1 = cd["RD"].get(1, []) if side == "buy" else cd["RU"].get(1, [])
    adv1 = cd["RU"].get(1, []) if side == "buy" else cd["RD"].get(1, [])
    if not fav1:
        return {r: 0 for r in range(1, tau_max + 1)}
    w = float(np.mean(adv1)) if adv1 else 0.0
    max_ell = int(max(fav1))
    v_prev = 0.0  # V(0): forced market at deadline, 0 improvement vs the then-current price
    sched: dict[int, int] = {}
    for r in range(1, tau_max + 1):
        best_ell, best_val = 0, -np.inf
        for ell in range(0, max_ell + 1):
            q = _survival(fav1, ell)
            val = q * ell + (1.0 - q) * (v_prev - w)
            if val > best_val:
                best_val, best_ell = val, ell
        sched[r] = best_ell
        v_prev = best_val
    return sched


def simulate_dynamic(
    side: Side, sched: dict[int, int], o, h, low, c, arrival: float, tick: float, tau_max: int
) -> tuple[float, int]:
    """Walk the window's bars; post sched[remaining] each minute; market at the deadline.

    Returns (shortfall_ticks vs arrival, fill_minute) where fill_minute = -1 if marketed.
    """
    n = len(o)
    for i in range(n):
        r = n - i  # minutes remaining including this one
        ell = sched.get(min(r, tau_max), 0)
        oi = float(o[i])
        if side == "buy":
            level = oi - ell * tick
            if float(low[i]) <= level:
                return (arrival - level) / tick, i
        else:
            level = oi + ell * tick
            if float(h[i]) >= level:
                return (level - arrival) / tick, i
    price = float(c[-1])  # deadline: market at the last close
    short = (arrival - price) / tick if side == "buy" else (price - arrival) / tick
    return short, -1


def evaluate_agent_dynamic(
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
    target_adaptive: float | None = None,
    train_end: pd.Timestamp | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compare single / blend / cutoff / adaptive / dp on the SAME orders (one pass).

    single/blend/cutoff reuse the slicing fill functions; adaptive/dp simulate the dynamic
    policies built from per-cell horizon tables. Returns {scheme: {"shortfall": ndarray}}.
    """
    target_adaptive = fill_rate_target if target_adaptive is None else target_adaptive
    names = ["single", "blend", "cutoff", "adaptive", "dp"]
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
    opens = df_ohlcv["open"].to_numpy()
    highs = df_ohlcv["high"].to_numpy()
    lows = df_ohlcv["low"].to_numpy()
    closes = df_ohlcv["close"].to_numpy()
    tau_ns = int(pd.Timedelta(minutes=tau).value)

    tables = fit_horizon_tables(
        t_ns_arr, ev, er, dx, sv, sr, sd, M, N, K, j_start, train_count,
        idx_ns, opens, highs, lows, closes, tau, tau_ns, tick,
    )
    sched_cache: dict[tuple[Cell, Side], tuple[dict, dict]] = {}
    out: dict[str, list[float]] = {nm: [] for nm in names}

    for t, row in trades.iterrows():
        side: Side = row["side"]
        t_ns = int(pd.Timestamp(t).value)
        w = bisect_right_local(t_ns_arr, t_ns - tau_ns) - 1
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
        cell: Cell = (m, n, k)
        epdf = (counts_RU if side == "sell" else counts_RD).get(cell, Counter())
        if not epdf or cell not in tables:
            continue  # keep all schemes on the same decision set
        ell_star = pick_ell_star(epdf, fill_rate_target)

        bars = _window_bars(idx_ns, opens, highs, lows, closes, t_ns, tau_ns)
        if bars is None:
            continue
        o, h, low_arr, c = bars
        open_j = float(o[0])  # basis-immune benchmark: the OHLC window open, not the agent price

        # static schemes (reuse slicing fills) -> shortfall vs the window open
        for nm, fn in (("single", fill_single), ("blend", fill_blend), ("cutoff", fill_cutoff)):
            price, _ = fn(side, ell_star, o, h, low_arr, c, tick)
            short = (price - open_j) / tick if side == "sell" else (open_j - price) / tick
            out[nm].append(short)

        # dynamic schemes (simulate_dynamic measures improvement vs its `arrival` arg = open_j)
        key = (cell, side)
        if key not in sched_cache:
            cd = tables[cell]
            sched_cache[key] = (
                adaptive_schedule(cd, side, target_adaptive, tau),
                dp_schedule(cd, side, tau),
            )
        adp, dpp = sched_cache[key]
        out["adaptive"].append(simulate_dynamic(side, adp, o, h, low_arr, c, open_j, tick, tau)[0])
        out["dp"].append(simulate_dynamic(side, dpp, o, h, low_arr, c, open_j, tick, tau)[0])

    return {nm: {"shortfall": np.array(out[nm], dtype=float)} for nm in names}


def bisect_right_local(arr: np.ndarray, value: int) -> int:
    """np.searchsorted right — local alias to keep the import surface explicit."""
    return int(np.searchsorted(arr, value, side="right"))
