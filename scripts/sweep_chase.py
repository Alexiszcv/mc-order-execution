"""Compare execution variants on the same no-lookahead (v2) decision path.

The static strategy picks a passive limit ℓ* ticks from the window open and, if it
doesn't fill, chases at the window close. The headline problem is that this chase tail
eats the median edge. This script holds the *decision* path fixed and swaps in
alternative executions / pickers, then reports fill rate and the 5th-percentile tail so
the trade-offs are visible:

  * baseline   — target picker (fill_rate_target), chase at close   [reproduces v2]
  * mid-chase  — target picker, chase at the window mid (H+L)/2
  * early-chase— target picker, bail intrabar once price runs TRIGGER ticks against us
  * cost-aware — expected-cost picker (no fixed target), chase at close

Why a self-contained simulator here instead of reusing `run_backtest`? Stream D owns
strategy + scripts and must NOT edit `backtest.py` (Stream C owns it), and the chase /
early-bail logic has to sit inside the per-window loop. To prove this loop is the *same*
decision path as the trustworthy v2 backtest, `baseline` is asserted to reproduce
`run_backtest_rolling` exactly (`_assert_matches_v2`).

No-lookahead: the regime/threshold/ePDF streaming below is copied structurally from
`run_backtest_rolling` — at decision j everything is built from windows strictly before
j. Reading bars *inside* window j to settle the fill is outcome simulation, identical in
spirit to the baseline reading the window close.

Run: `python scripts/sweep_chase.py` from the repo root.
"""

from __future__ import annotations

import bisect
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from order_mgmt.backtest import run_backtest_rolling  # noqa: E402
from order_mgmt.pipeline import load_market_indexed  # noqa: E402
from order_mgmt.strategy import (  # noqa: E402
    chase_price,
    pick_ell_star,
    pick_ell_star_cost_aware,
    simulate_early_chase,
)
from order_mgmt.ticks import resolve_tick  # noqa: E402
from plot_volume import _compute_stats  # noqa: E402
from ranges import compute_all_ranges  # noqa: E402
from regime import compute_ewma_series  # noqa: E402

SEED = 0  # no RNG in this deterministic backtest; seed for reproducibility hygiene.

TAU = 5
HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200
FILL_RATE_TARGET = 0.6
TRIGGER_TICKS = 4  # early-chase: bail once price runs this many ticks against the limit.
CHASE_COST_TICKS = 2.0  # cost-aware picker: assumed adverse ticks paid on an unfill.

MARKETS = [
    ("Gold", "Gold"),
    ("Nasdaq", "Nasdaq"),
    ("Bunds", "Bunds"),
]

Side = Literal["buy", "sell"]
Picker = Literal["target", "cost"]
Chase = Literal["close", "mid", "early"]


@dataclass(frozen=True)
class VariantResult:
    n_decisions: int
    n_filled: int
    fill_rate: float
    avg: float
    median: float
    p05: float  # 5th-percentile slippage = the adverse tail we want to shrink.


def _state(value: float, sorted_arr: list[float], n_states: int) -> int | None:
    """Regime-state assignment — identical to `backtest._state` / `epdf.build_epdf`."""
    L = len(sorted_arr)
    if n_states > L:
        return None
    rank = bisect.bisect_left(sorted_arr, value)
    return min(rank * n_states // L + 1, n_states)


def _summarize(n_filled: int, slippage: list[float]) -> VariantResult:
    n_dec = len(slippage)
    return VariantResult(
        n_decisions=n_dec,
        n_filled=n_filled,
        fill_rate=n_filled / n_dec if n_dec else 0.0,
        avg=float(np.mean(slippage)) if slippage else 0.0,
        median=float(np.median(slippage)) if slippage else 0.0,
        p05=float(np.percentile(slippage, 5)) if slippage else 0.0,
    )


def run_rolling_variant(
    df_1min: pd.DataFrame,
    tau: int,
    tick: float,
    proper_days: list,
    side: Side,
    *,
    half_life: int,
    M: int,
    N: int,
    K: int,
    j_start: int,
    picker: Picker = "target",
    fill_rate_target: float = FILL_RATE_TARGET,
    chase_cost_ticks: float = CHASE_COST_TICKS,
    chase: Chase = "close",
    trigger_ticks: int = TRIGGER_TICKS,
) -> VariantResult:
    """No-lookahead rolling backtest with a pluggable picker and chase model.

    Structure mirrors `order_mgmt.backtest.run_backtest_rolling`; only the picker call
    and the execution/chase branch differ.
    """
    t_list, _ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(
        df_1min, tau, tick, proper_days
    )
    if len(t_list) <= j_start:
        return _summarize(0, [])

    ewma_range, ewma_vol = compute_ewma_series(t_list, _ell_r, vol_list, half_life)
    ev = np.asarray(ewma_vol, dtype=float)
    er = np.asarray(ewma_range, dtype=float)
    dx_arr = np.asarray(dx_list, dtype=float)

    # Per-window bar slices, for chase models that read the realized intrabar path.
    idx_ns = df_1min.index.as_unit("ns").asi8
    high_arr = df_1min["high"].to_numpy()
    low_arr = df_1min["low"].to_numpy()
    close_arr = df_1min["close"].to_numpy()
    open_arr = df_1min["open"].to_numpy()
    t_ns = np.array([t.value for t in t_list], dtype=np.int64)
    lo_idx = np.searchsorted(idx_ns, t_ns, side="left")  # window j == bars [lo, lo+tau)

    # Streaming regime state — strictly-before-j prefix, exactly like run_backtest_rolling.
    pre = max(0, j_start - 1)
    sv = sorted(float(v) for v in ev[:pre] if not np.isnan(v))
    sr = sorted(float(v) for v in er[:pre] if not np.isnan(v))
    sd = sorted(float(v) for v in dx_arr[:pre] if not np.isnan(v))
    counts: dict[tuple[int, int, int], Counter] = {}

    slippage: list[float] = []
    n_filled = 0

    for j in range(max(j_start, 1), len(t_list)):
        v_prev, r_prev, d_prev = ev[j - 1], er[j - 1], dx_arr[j - 1]
        cell: tuple[int, int, int] | None = None
        if not (np.isnan(v_prev) or np.isnan(r_prev) or np.isnan(d_prev)):
            m = _state(float(v_prev), sv, M)
            n = _state(float(r_prev), sr, N)
            k = _state(float(d_prev), sd, K)
            if m is not None and n is not None and k is not None:
                cell = (m, n, k)

        if cell is not None:
            epdf = counts.get(cell, Counter())
            if epdf:
                if picker == "cost":
                    ell_star = pick_ell_star_cost_aware(epdf, chase_cost_ticks)
                else:
                    ell_star = pick_ell_star(epdf, fill_rate_target)

                lo = int(lo_idx[j])
                hi = lo + tau
                open_j = float(open_arr[lo])
                close_j = float(close_arr[hi - 1])

                price, filled = _execute(
                    side, open_j, close_j, ell_star, tick,
                    ell_u[j], ell_d[j],
                    high_arr[lo:hi], low_arr[lo:hi], close_arr[lo:hi],
                    chase=chase, trigger_ticks=trigger_ticks,
                )
                if filled:
                    n_filled += 1
                slip = (price - open_j) / tick if side == "sell" else (open_j - price) / tick
                slippage.append(slip)

        if not np.isnan(v_prev):
            bisect.insort(sv, float(v_prev))
        if not np.isnan(r_prev):
            bisect.insort(sr, float(r_prev))
        if not np.isnan(d_prev):
            bisect.insort(sd, float(d_prev))

        if cell is not None:
            counts.setdefault(cell, Counter())[ell_u[j] if side == "sell" else ell_d[j]] += 1

    return _summarize(n_filled, slippage)


def _execute(
    side: Side,
    open_j: float,
    close_j: float,
    ell_star: int,
    tick: float,
    ell_u_j: int,
    ell_d_j: int,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    chase: Chase,
    trigger_ticks: int,
) -> tuple[float, bool]:
    """Resolve one window's execution price + fill flag under the chosen chase model."""
    if chase == "early":
        return simulate_early_chase(
            side, open_j, ell_star, tick, highs, lows, closes, trigger_ticks=trigger_ticks
        )

    # close / mid: window-level fill check (same as run_backtest_rolling), differing
    # only in where we get done on an unfill.
    realized_ell = ell_u_j if side == "sell" else ell_d_j
    if realized_ell >= ell_star:
        price = open_j + ell_star * tick if side == "sell" else open_j - ell_star * tick
        return price, True
    if chase == "mid":
        return chase_price(
            open_j, tick, policy="mid",
            window_high=float(highs.max()), window_low=float(lows.min()),
        ), False
    return chase_price(open_j, tick, policy="close", window_close=close_j), False


def _assert_matches_v2(df_ohlcv, tick, proper_days, side: Side) -> None:
    """Guard: baseline (target picker + close chase) must equal `run_backtest_rolling`.

    Catches any drift between this self-contained loop and Stream C's backtest.
    """
    mine = run_rolling_variant(
        df_ohlcv, TAU, tick, proper_days, side,
        half_life=HALF_LIFE, M=M, N=N, K=K, j_start=J_START,
        picker="target", fill_rate_target=FILL_RATE_TARGET, chase="close",
    )
    theirs = run_backtest_rolling(
        df_ohlcv, tau=TAU, tick=tick, proper_days=proper_days, side=side,
        fill_rate_target=FILL_RATE_TARGET, half_life=HALF_LIFE,
        M=M, N=N, K=K, j_start=J_START,
    )
    assert mine.n_decisions == theirs.n_decisions, (mine.n_decisions, theirs.n_decisions)
    assert mine.n_filled == theirs.n_filled, (mine.n_filled, theirs.n_filled)
    assert abs(mine.avg - theirs.avg_slippage_ticks) < 1e-9, (mine.avg, theirs.avg_slippage_ticks)


def _find_market_dir(substring: str) -> Path | None:
    data = ROOT / "data"
    if not data.exists():
        return None
    for p in data.iterdir():
        if p.is_dir() and substring.lower() in p.name.lower():
            return p
    return None


def sweep_one_market(substring: str, label: str) -> None:
    market_dir = _find_market_dir(substring)
    if market_dir is None:
        print(f"[{label}] data dir not found")
        return
    df = load_market_indexed(market_dir)
    if df.empty:
        print(f"[{label}] no usable bars")
        return

    first_stem = df["contract"].unique().tolist()[0]
    df_ohlcv = df[["open", "high", "low", "close", "volume"]]
    _, inferred_tick, proper_days, n_green, _ = _compute_stats(df_ohlcv)
    tick = resolve_tick(first_stem, inferred_tick)
    print(f"\n=== {label} (tick={tick:g}, {n_green} active days) ===")
    print(f"{'side':4} {'variant':11} {'n':>6} {'fill':>6} {'avg':>8} {'median':>8} {'p05(tail)':>10}")
    print("-" * 58)

    for side in ("buy", "sell"):
        _assert_matches_v2(df_ohlcv, tick, proper_days, side)  # baseline == v2
        variants = {
            "baseline": dict(picker="target", chase="close"),
            "mid-chase": dict(picker="target", chase="mid"),
            f"early(>{TRIGGER_TICKS}t)": dict(picker="target", chase="early"),
            f"cost(c={CHASE_COST_TICKS:g})": dict(picker="cost", chase="close"),
        }
        for name, kw in variants.items():
            r = run_rolling_variant(
                df_ohlcv, TAU, tick, proper_days, side,
                half_life=HALF_LIFE, M=M, N=N, K=K, j_start=J_START,
                fill_rate_target=FILL_RATE_TARGET, chase_cost_ticks=CHASE_COST_TICKS,
                trigger_ticks=TRIGGER_TICKS, **kw,
            )
            print(
                f"{side:4} {name:11} {r.n_decisions:>6} {r.fill_rate:>5.1%} "
                f"{r.avg:>+8.3f} {r.median:>+8.3f} {r.p05:>+10.2f}"
            )


def main() -> None:
    np.random.seed(SEED)
    print(
        "Comparing execution variants on the fixed v2 decision path.\n"
        "p05 = 5th-percentile slippage (the adverse tail); higher (less negative) is better.\n"
        f"baseline reproduces run_backtest_rolling (asserted). trigger={TRIGGER_TICKS}t, "
        f"chase_cost={CHASE_COST_TICKS:g}t."
    )
    for substring, label in MARKETS:
        sweep_one_market(substring, label)


if __name__ == "__main__":
    main()
