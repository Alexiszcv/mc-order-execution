"""Execution-value metrics for the AI agent's trades.

What this measures — and what it does NOT
------------------------------------------
There is NO order-book / quote data, so this does not and cannot model bid-ask
spread cost or market impact. It measures **implementation shortfall**: the
realized execution price vs the agent's arrival (decision) price, in ticks,
under the project's regime-conditioned limit-order policy.

Mechanism (reuses the backtest's fill model verbatim):
  - The agent decides to trade at time t (arrival price P_t).
  - We post a passive limit ℓ* ticks on the favorable side of the execution
    window's open. Using only OHLC we know whether price traversed that level
    inside [t, t+τ): if the realized half-range reaches ℓ* (R_U≥ℓ* for a sell,
    R_D≥ℓ* for a buy) the limit fills ℓ* ticks better; otherwise we chase at the
    window close and may do worse.

Sign convention (matches `backtest._simulate_fill`): positive ticks = the
strategy BEAT the benchmark (sold higher / bought lower).

CAVEAT: the fill model is optimistic — touching the limit level is assumed to
fill (no queue position, no partial fills, no spread). Mean shortfall is dragged
toward zero by the chase tail, so the MEDIAN is the honest headline.

No-lookahead
------------
ePDFs + regime quantile thresholds are fit on OHLC windows strictly before
`train_end` (default: the first agent decision). Every decision then reads only
that frozen model plus bars with timestamp < t. See the INVARIANT comments below.
"""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from order_mgmt.agent.loader import AgentSeries, trade_decisions
from order_mgmt.backtest import _final_sorted, _simulate_fill, _state
from order_mgmt.baselines import Side
from order_mgmt.strategy import pick_ell_star
from ranges import compute_all_ranges
from regime import compute_ewma_series


@dataclass(frozen=True)
class DecisionFill:
    t: pd.Timestamp
    side: Side
    qty: int
    arrival_price: float  # agent's decision price = implementation-shortfall benchmark
    open_j: float  # execution-window open = backtest slippage benchmark
    realized_price: float
    filled: bool
    ell_star: int
    slippage_ticks: float  # signed vs window open (positive = beat open)
    shortfall_ticks: float  # signed vs arrival price (positive = beat arrival)
    fill_prob: float  # ex-ante P(R≥ℓ*) in the regime cell — the model's fill confidence
    vol_proxy: float  # regime EWMA range level (a volatility proxy), known at decision time


# Order-sizing rules: the agent sets side+timing; OUR MODEL can also set the size.
# All weights are ex-ante (computed from info known at the decision) so applying them to
# the realized shortfall introduces no lookahead.
SizingRule = str  # "agent" | "confidence" | "inverse_vol"


def size_weights(fills: list[DecisionFill], rule: SizingRule) -> np.ndarray:
    """Per-decision size weight under a sizing rule.

    - "agent"       : the agent's own |Δposition| (baseline — model sizes nothing).
    - "confidence"  : fill_prob · ℓ* = expected ticks captured (size ∝ the model's edge).
    - "inverse_vol" : 1/σ̄ proxy (smaller size in volatile regimes; risk-based).
    """
    if rule == "agent":
        return np.array([float(f.qty) for f in fills], dtype=float)
    if rule == "confidence":
        return np.array([f.fill_prob * f.ell_star for f in fills], dtype=float)
    if rule == "inverse_vol":
        return np.array([1.0 / f.vol_proxy if f.vol_proxy > 0 else 0.0 for f in fills], dtype=float)
    raise ValueError(f"unknown sizing rule: {rule!r}")


def size_weighted_shortfall(fills: list[DecisionFill], rule: SizingRule) -> float:
    """Size-weighted mean shortfall in ticks (Σ w·shortfall / Σ w). Comparable across rules."""
    if not fills:
        return 0.0
    w = size_weights(fills, rule)
    s = np.array([f.shortfall_ticks for f in fills], dtype=float)
    wsum = w.sum()
    return float((w * s).sum() / wsum) if wsum > 0 else 0.0


@dataclass(frozen=True)
class AgentEvalResult:
    market: str
    n_decisions: int
    n_filled: int
    fill_rate: float
    mean_shortfall_ticks: float
    median_shortfall_ticks: float
    mean_slippage_ticks: float
    median_slippage_ticks: float
    captured_improvement_notional: float  # Σ shortfall·tick·qty (captured improvement, NOT impact cost)
    unfilled_tail_cost_ticks: float  # mean shortfall on the unfilled (chased) subset
    value_add_vs_market_ticks: float  # mean shortfall (market-on-decision ref = 0)
    value_add_vs_vwap_ticks: float  # mean(shortfall) − mean(vwap shortfall)
    fills: list[DecisionFill]


def _window_stats(
    idx_ns: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    typ: np.ndarray,
    vol: np.ndarray,
    t_ns: int,
    tau_ns: int,
) -> tuple[float, float, float, float, float] | None:
    """OHLC + VWAP over the bars in [t, t+τ). None if the interval has no bars.

    Located by searchsorted on the int64-ns index (O(log n) per decision), the same
    technique `baselines.vwap_baseline` uses. Gaps are handled naturally — whatever
    bars exist in the interval define the window.
    """
    lo = int(np.searchsorted(idx_ns, t_ns, side="left"))
    hi = int(np.searchsorted(idx_ns, t_ns + tau_ns, side="left"))
    if hi <= lo:
        return None
    open_j = float(opens[lo])
    high_j = float(highs[lo:hi].max())
    low_j = float(lows[lo:hi].min())
    close_j = float(closes[hi - 1])
    w_typ = typ[lo:hi]
    w_vol = vol[lo:hi]
    vsum = w_vol.sum()
    vwap_j = float(w_typ.mean()) if vsum == 0 else float((w_typ * w_vol).sum() / vsum)
    return open_j, high_j, low_j, close_j, vwap_j


def _window_bars(
    idx_ns: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    t_ns: int,
    tau_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """The 1-min bar arrays (o, h, l, c) falling in [t, t+τ). None if the interval is empty."""
    lo = int(np.searchsorted(idx_ns, t_ns, side="left"))
    hi = int(np.searchsorted(idx_ns, t_ns + tau_ns, side="left"))
    if hi <= lo:
        return None
    return opens[lo:hi], highs[lo:hi], lows[lo:hi], closes[lo:hi]


def _fit_regime_model(
    ell_u: list[int],
    ell_d: list[int],
    ev: np.ndarray,
    er: np.ndarray,
    dx: np.ndarray,
    train_count: int,
    M: int,
    N: int,
    K: int,
    j_start: int,
) -> tuple[dict, dict, list[float], list[float], list[float]]:
    """Fit per-regime R_U/R_D counts on the training windows with FROZEN thresholds.

    Thresholds are the full-training quantiles (via `_final_sorted`), and the SAME
    frozen sorted lists are used both here (to bin each training outcome) and later
    to classify each decision — so build and lookup share one cell definition (the
    consistency property the v1 backtest relies on). Causal because every training
    window precedes `train_end` (and hence every decision).
    """
    sv = _final_sorted(ev[:train_count])
    sr = _final_sorted(er[:train_count])
    sd = _final_sorted(dx[:train_count])

    counts_RU: dict[tuple[int, int, int], Counter] = {}
    counts_RD: dict[tuple[int, int, int], Counter] = {}
    for j in range(max(j_start, 1), train_count):
        v_prev, r_prev, d_prev = ev[j - 1], er[j - 1], dx[j - 1]
        if np.isnan(v_prev) or np.isnan(r_prev) or np.isnan(d_prev):
            continue
        m = _state(float(v_prev), sv, M)
        n = _state(float(r_prev), sr, N)
        k = _state(float(d_prev), sd, K)
        if m is None or n is None or k is None:
            continue
        key = (m, n, k)
        if key not in counts_RU:
            counts_RU[key] = Counter()
            counts_RD[key] = Counter()
        counts_RU[key][ell_u[j]] += 1
        counts_RD[key][ell_d[j]] += 1
    return counts_RU, counts_RD, sv, sr, sd


def evaluate_agent_execution(
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
    train_end: pd.Timestamp | None = None,
) -> AgentEvalResult:
    """Evaluate the regime-conditioned execution of the agent's trades on OHLC data."""
    trades = trade_decisions(agent)

    # Range + EWMA passes over the full OHLC history (computed once).
    t_list, ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(
        df_ohlcv, tau, tick, proper_days
    )
    if not t_list or trades.empty:
        return _empty_result(agent.market)
    ewma_range, ewma_vol = compute_ewma_series(t_list, ell_r, vol_list, half_life)
    ev = np.asarray(ewma_vol, dtype=float)
    er = np.asarray(ewma_range, dtype=float)
    dx = np.asarray(dx_list, dtype=float)

    # INVARIANT: ePDFs/regime thresholds see only OHLC windows with start time < train_end.
    if train_end is None:
        train_end = trades.index.min()
    t_ns_arr = np.asarray([pd.Timestamp(t).value for t in t_list], dtype=np.int64)
    train_count = int(np.searchsorted(t_ns_arr, pd.Timestamp(train_end).value, side="left"))

    counts_RU, counts_RD, sv, sr, sd = _fit_regime_model(
        ell_u, ell_d, ev, er, dx, train_count, M, N, K, j_start
    )

    # Precompute OHLC arrays for the per-decision window lookup.
    if not df_ohlcv.index.is_monotonic_increasing:
        df_ohlcv = df_ohlcv.sort_index()
    idx_ns = df_ohlcv.index.as_unit("ns").asi8
    opens = df_ohlcv["open"].to_numpy()
    highs = df_ohlcv["high"].to_numpy()
    lows = df_ohlcv["low"].to_numpy()
    closes = df_ohlcv["close"].to_numpy()
    typ = ((df_ohlcv["high"] + df_ohlcv["low"] + df_ohlcv["close"]) / 3.0).to_numpy()
    vol = df_ohlcv["volume"].to_numpy()
    tau_ns = int(pd.Timedelta(minutes=tau).value)

    fills: list[DecisionFill] = []
    vwap_shortfalls: list[float] = []

    for t, row in trades.iterrows():
        side: Side = row["side"]
        qty = int(row["qty"])
        arrival = float(row["price"])
        t_ns = int(pd.Timestamp(t).value)

        # INVARIANT: read regime only from the latest OHLC window CLOSED before t,
        # i.e. window w with t_list[w] + τ <= t. EWMA[w] uses data up to window w only.
        w = bisect.bisect_right(t_ns_arr, t_ns - tau_ns) - 1
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

        epdf = (counts_RU if side == "sell" else counts_RD).get((m, n, k), Counter())
        if not epdf:
            continue
        ell_star = pick_ell_star(epdf, fill_rate_target)
        # Ex-ante fill confidence P(R≥ℓ*) in this cell (known at decision → no lookahead).
        total = sum(epdf.values())
        fill_prob = sum(c for ell, c in epdf.items() if ell >= ell_star) / total if total else 0.0

        stats = _window_stats(idx_ns, opens, highs, lows, closes, typ, vol, t_ns, tau_ns)
        if stats is None:
            continue
        open_j, high_j, low_j, close_j, vwap_j = stats
        ell_u_j = max(round((high_j - open_j) / tick), 0)
        ell_d_j = max(round((open_j - low_j) / tick), 0)

        price, filled, slip = _simulate_fill(
            side, ell_star, ell_u_j, ell_d_j, open_j, close_j, tick
        )
        # Benchmark vs the OHLC window OPEN (basis-immune): the agent's CSV price sits on a
        # different contract at rolls, so it is NOT a valid reference. The open is the price
        # on the instrument we actually fill — the standard implementation-shortfall arrival.
        if side == "sell":
            shortfall = (price - open_j) / tick
            vwap_short = (vwap_j - open_j) / tick
        else:
            shortfall = (open_j - price) / tick
            vwap_short = (open_j - vwap_j) / tick

        fills.append(
            DecisionFill(
                t=pd.Timestamp(t),
                side=side,
                qty=qty,
                arrival_price=arrival,
                open_j=open_j,
                realized_price=price,
                filled=filled,
                ell_star=ell_star,
                slippage_ticks=slip,
                shortfall_ticks=shortfall,
                fill_prob=fill_prob,
                vol_proxy=float(r_prev),
            )
        )
        vwap_shortfalls.append(vwap_short)

    return _aggregate(agent.market, fills, vwap_shortfalls, tick)


def evaluate_agent_schemes(
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
    train_end: pd.Timestamp | None = None,
    schemes: dict | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Run EVERY order-slicing scheme on the SAME orders / regime / ℓ*, one pass.

    Returns {scheme_name: {"shortfall": ndarray, "fill_frac": ndarray}}. The regime fit,
    ℓ* and window bars are shared across schemes so the only thing that differs is HOW the
    order is cut — making the comparison apples-to-apples. Same no-lookahead invariant as
    `evaluate_agent_execution` (it is the single-shot scheme here, cross-checked in tests).
    """
    from order_mgmt.agent.slicing import SCHEMES

    schemes = schemes or SCHEMES
    empty = {name: {"shortfall": np.array([]), "fill_frac": np.array([])} for name in schemes}
    trades = trade_decisions(agent)

    t_list, ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(
        df_ohlcv, tau, tick, proper_days
    )
    if not t_list or trades.empty:
        return empty
    ewma_range, ewma_vol = compute_ewma_series(t_list, ell_r, vol_list, half_life)
    ev = np.asarray(ewma_vol, dtype=float)
    er = np.asarray(ewma_range, dtype=float)
    dx = np.asarray(dx_list, dtype=float)

    # INVARIANT: ePDFs/regime thresholds see only OHLC windows with start time < train_end.
    if train_end is None:
        train_end = trades.index.min()
    t_ns_arr = np.asarray([pd.Timestamp(t).value for t in t_list], dtype=np.int64)
    train_count = int(np.searchsorted(t_ns_arr, pd.Timestamp(train_end).value, side="left"))
    counts_RU, counts_RD, sv, sr, sd = _fit_regime_model(
        ell_u, ell_d, ev, er, dx, train_count, M, N, K, j_start
    )

    if not df_ohlcv.index.is_monotonic_increasing:
        df_ohlcv = df_ohlcv.sort_index()
    idx_ns = df_ohlcv.index.as_unit("ns").asi8
    opens = df_ohlcv["open"].to_numpy()
    highs = df_ohlcv["high"].to_numpy()
    lows = df_ohlcv["low"].to_numpy()
    closes = df_ohlcv["close"].to_numpy()
    tau_ns = int(pd.Timedelta(minutes=tau).value)

    out: dict[str, dict[str, list]] = {name: {"shortfall": [], "fill_frac": []} for name in schemes}

    for t, row in trades.iterrows():
        side: Side = row["side"]
        t_ns = int(pd.Timestamp(t).value)

        # INVARIANT: regime read from the latest OHLC window CLOSED before t.
        w = bisect.bisect_right(t_ns_arr, t_ns - tau_ns) - 1
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
        epdf = (counts_RU if side == "sell" else counts_RD).get((m, n, k), Counter())
        if not epdf:
            continue
        ell_star = pick_ell_star(epdf, fill_rate_target)

        bars = _window_bars(idx_ns, opens, highs, lows, closes, t_ns, tau_ns)
        if bars is None:
            continue
        o, h, low_arr, c = bars
        open_j = float(o[0])  # basis-immune benchmark (see evaluate_agent_execution)
        for name, (fn, kw) in schemes.items():
            price, frac = fn(side, ell_star, o, h, low_arr, c, tick, **kw)
            short = (price - open_j) / tick if side == "sell" else (open_j - price) / tick
            out[name]["shortfall"].append(short)
            out[name]["fill_frac"].append(frac)

    return {
        name: {
            "shortfall": np.array(d["shortfall"], dtype=float),
            "fill_frac": np.array(d["fill_frac"], dtype=float),
        }
        for name, d in out.items()
    }


def _empty_result(market: str) -> AgentEvalResult:
    return AgentEvalResult(market, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])


def _aggregate(
    market: str,
    fills: list[DecisionFill],
    vwap_shortfalls: list[float],
    tick: float,
) -> AgentEvalResult:
    if not fills:
        return _empty_result(market)
    short = np.array([f.shortfall_ticks for f in fills], dtype=float)
    slip = np.array([f.slippage_ticks for f in fills], dtype=float)
    filled_mask = np.array([f.filled for f in fills], dtype=bool)
    qty = np.array([f.qty for f in fills], dtype=float)
    vwap_short = np.array(vwap_shortfalls, dtype=float)

    n_filled = int(filled_mask.sum())
    mean_short = float(short.mean())
    unfilled = short[~filled_mask]
    return AgentEvalResult(
        market=market,
        n_decisions=len(fills),
        n_filled=n_filled,
        fill_rate=n_filled / len(fills),
        mean_shortfall_ticks=mean_short,
        median_shortfall_ticks=float(np.median(short)),
        mean_slippage_ticks=float(slip.mean()),
        median_slippage_ticks=float(np.median(slip)),
        captured_improvement_notional=float((short * tick * qty).sum()),
        unfilled_tail_cost_ticks=float(unfilled.mean()) if unfilled.size else 0.0,
        value_add_vs_market_ticks=mean_short,  # market-on-decision shortfall ≡ 0
        value_add_vs_vwap_ticks=mean_short - float(vwap_short.mean()),
        fills=fills,
    )
