"""Ablation: does the 27-cell regime conditioning actually earn its keep?

The strategy conditions the limit distance ℓ* on a (volume, range, Δx) regime cell. This
script asks whether that machinery beats trivial alternatives, by scoring a *ladder* of
ℓ*-rules on the **same decision windows with the same fills** so the means are comparable:

  * conditional   — pick_ell_star on the 27-cell ePDF              (the thing under test)
  * unconditional — pick_ell_star on ONE global ePDF (M=N=K=1)     (does splitting help?)
  * persistence   — ℓ* = the previous window's realized range      (beat a 1-lag carry?)
  * random        — ℓ* uniform in the cell's plausible band        (beat noise?)
  * VWAP          — volume-weighted execution price (reference; not a limit rule)

Why a self-contained loop instead of calling run_backtest five times? Two reasons. (1) The
rungs must share an identical decision set — run_backtest_rolling(M=1) would act on *more*
windows than M=3 (its single cell is never empty), so its mean would not be comparable.
Gating every rung on the *conditional* cell being non-empty fixes that. (2) Stream D must
not edit backtest.py (Stream C). The loop below is copied structurally from
run_backtest_rolling; to prove it hasn't drifted, the conditional rung is asserted to
reproduce run_backtest_rolling exactly (`_assert_matches_v2`) — the same guard sweep_chase.py
uses.

No-lookahead: every ℓ* reads only data strictly before window j. persistence's j-1 window is
strictly earlier (windows are non-overlapping, ranges.py). Reading window-j bars to settle the
fill / VWAP is outcome simulation, identical in spirit to the baseline reading the window close.

Run: `python scripts/ablation.py` from the repo root (5 markets × τ{5,10,15,30,60}; a few minutes).
Outputs `reports/figures/ablation_pareto_<market>.png`, `ablation_tau_<market>.png`, and a table.
"""

from __future__ import annotations

import bisect
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from order_mgmt.backtest import run_backtest_rolling  # noqa: E402
from order_mgmt.pipeline import load_market_indexed  # noqa: E402
from order_mgmt.strategy import pick_ell_star, pick_ell_star_random  # noqa: E402
from order_mgmt.ticks import resolve_tick  # noqa: E402
from plot_volume import _compute_stats  # noqa: E402
from ranges import compute_all_ranges  # noqa: E402
from regime import compute_ewma_series  # noqa: E402

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 0  # fresh default_rng(SEED) per (market, side, τ) → each config reproducible.

HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200
TARGETS = [0.4, 0.5, 0.6, 0.7, 0.8]
ASSERT_TARGET = 0.6  # the target at which conditional must reproduce run_backtest_rolling.
TAUS = [5, 10, 15, 30, 60]
TAU_FOR_PARETO = 5  # the Pareto figure is drawn at this τ.
TARGET_FOR_TAU = 0.6  # the τ-sensitivity figure fixes this target.

# Picker rungs are target-dependent; these are not (single point on the Pareto).
FLAT_RUNGS = ("persistence", "random", "vwap")

MARKETS = [
    ("Gold", "Gold"),
    ("Nasdaq", "Nasdaq"),
    ("Bunds", "Bunds"),
    ("EuroStoxx", "EuroStoxx"),
    ("GBP", "GBP"),  # carries the existing tiny-tick data-quality caveat (notes/strategy-sweep.md).
]

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class RungResult:
    n_decisions: int
    n_filled: int
    fill_rate: float
    avg: float
    median: float
    p05: float
    slips: list[float]  # per-decision signed slippage (ticks) — kept for invariant tests.


def _state(value: float, sorted_arr: list[float], n_states: int) -> int | None:
    """Regime-state assignment — identical to `backtest._state` / `epdf.build_epdf`."""
    L = len(sorted_arr)
    if n_states > L:
        return None
    rank = bisect.bisect_left(sorted_arr, value)
    return min(rank * n_states // L + 1, n_states)


def _summarize(n_filled: int, slips: list[float]) -> RungResult:
    n = len(slips)
    return RungResult(
        n_decisions=n,
        n_filled=n_filled,
        fill_rate=n_filled / n if n else 0.0,
        avg=float(np.mean(slips)) if slips else 0.0,
        median=float(np.median(slips)) if slips else 0.0,
        p05=float(np.percentile(slips, 5)) if slips else 0.0,
        slips=list(slips),
    )


def _score(
    side: Side,
    ell_star: int,
    ell_u_j: int,
    ell_d_j: int,
    open_j: float,
    close_j: float,
    tick: float,
) -> tuple[float, bool]:
    """Same fill mechanics as run_backtest_rolling: fill at the limit iff the realized
    range reaches it, else chase at the window close."""
    if side == "sell":
        if ell_u_j >= ell_star:
            return open_j + ell_star * tick, True
        return close_j, False
    if ell_d_j >= ell_star:
        return open_j - ell_star * tick, True
    return close_j, False


def _slip(side: Side, price: float, open_j: float, tick: float) -> float:
    return (price - open_j) / tick if side == "sell" else (open_j - price) / tick


def run_ablation(
    df_1min,
    tau: int,
    tick: float,
    proper_days: list,
    side: Side,
    *,
    half_life: int = HALF_LIFE,
    m_states: int = M,
    n_states: int = N,
    k_states: int = K,
    j_start: int = J_START,
    targets: list[float] = TARGETS,
) -> dict[str, object]:
    """One no-lookahead streaming pass scoring every rung on the SAME decision set.

    Returns {"conditional": {target: RungResult}, "unconditional": {target: RungResult},
             "persistence": RungResult, "random": RungResult, "vwap": RungResult}.
    """
    t_list, _ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(
        df_1min, tau, tick, proper_days
    )
    empty = _summarize(0, [])
    if len(t_list) <= j_start:
        return {
            "conditional": {t: empty for t in targets},
            "unconditional": {t: empty for t in targets},
            "persistence": empty,
            "random": empty,
            "vwap": empty,
        }

    ewma_range, ewma_vol = compute_ewma_series(t_list, _ell_r, vol_list, half_life)
    ev = np.asarray(ewma_vol, dtype=float)
    er = np.asarray(ewma_range, dtype=float)
    dx_arr = np.asarray(dx_list, dtype=float)

    # Per-window bar slices, used only for the VWAP reference (volume-weighted typical price).
    idx_ns = df_1min.index.as_unit("ns").asi8
    high_arr = df_1min["high"].to_numpy()
    low_arr = df_1min["low"].to_numpy()
    close_arr = df_1min["close"].to_numpy()
    vol_arr = df_1min["volume"].to_numpy()
    t_ns = np.array([t.value for t in t_list], dtype=np.int64)
    lo_idx = np.searchsorted(idx_ns, t_ns, side="left")  # window j == bars [lo, lo+tau)

    opens_series = df_1min["open"]
    closes_series = df_1min["close"]
    dt_last = pd.Timedelta(minutes=tau - 1)

    # Streaming regime state — strictly-before-j prefix, exactly like run_backtest_rolling.
    pre = max(0, j_start - 1)
    sv = sorted(float(v) for v in ev[:pre] if not np.isnan(v))
    sr = sorted(float(v) for v in er[:pre] if not np.isnan(v))
    sd = sorted(float(v) for v in dx_arr[:pre] if not np.isnan(v))
    counts: dict[tuple[int, int, int], Counter] = {}
    counts_global: Counter = Counter()  # the M=N=K=1 ePDF — pooled over all cells.

    rng = np.random.default_rng(SEED)
    cond_slip: dict[float, list[float]] = {t: [] for t in targets}
    cond_fill: dict[float, int] = {t: 0 for t in targets}
    uncond_slip: dict[float, list[float]] = {t: [] for t in targets}
    uncond_fill: dict[float, int] = {t: 0 for t in targets}
    pers_slip: list[float] = []
    pers_fill = 0
    rand_slip: list[float] = []
    rand_fill = 0
    vwap_slip: list[float] = []

    for j in range(max(j_start, 1), len(t_list)):
        v_prev, r_prev, d_prev = ev[j - 1], er[j - 1], dx_arr[j - 1]
        cell: tuple[int, int, int] | None = None
        if not (np.isnan(v_prev) or np.isnan(r_prev) or np.isnan(d_prev)):
            mm = _state(float(v_prev), sv, m_states)
            nn = _state(float(r_prev), sr, n_states)
            kk = _state(float(d_prev), sd, k_states)
            if mm is not None and nn is not None and kk is not None:
                cell = (mm, nn, kk)

        if cell is not None:
            epdf = counts.get(cell, Counter())
            if epdf:  # decision gate — identical to run_backtest_rolling.
                t = t_list[j]
                try:
                    open_j = float(opens_series.loc[t])
                    close_j = float(closes_series.loc[t + dt_last])
                except KeyError:
                    open_j = close_j = None

                if open_j is not None:
                    eu, ed = ell_u[j], ell_d[j]
                    # --- target-dependent rungs ---
                    for tgt in targets:
                        ell_c = pick_ell_star(epdf, tgt)
                        price, filled = _score(side, ell_c, eu, ed, open_j, close_j, tick)
                        cond_slip[tgt].append(_slip(side, price, open_j, tick))
                        cond_fill[tgt] += int(filled)

                        ell_u0 = pick_ell_star(counts_global, tgt)
                        price, filled = _score(side, ell_u0, eu, ed, open_j, close_j, tick)
                        uncond_slip[tgt].append(_slip(side, price, open_j, tick))
                        uncond_fill[tgt] += int(filled)
                    # --- target-independent rungs (computed once per decision) ---
                    # persistence: previous window's realized range on our side (no-lookahead,
                    # j-1 is strictly earlier). Clamp ≥ 0 (already non-negative from ranges.py).
                    ell_p = max(0, int(ell_u[j - 1] if side == "sell" else ell_d[j - 1]))
                    price, filled = _score(side, ell_p, eu, ed, open_j, close_j, tick)
                    pers_slip.append(_slip(side, price, open_j, tick))
                    pers_fill += int(filled)

                    ell_rd = pick_ell_star_random(epdf, rng)
                    price, filled = _score(side, ell_rd, eu, ed, open_j, close_j, tick)
                    rand_slip.append(_slip(side, price, open_j, tick))
                    rand_fill += int(filled)

                    # VWAP reference: volume-weighted typical price over the window bars
                    # (formula matches baselines.vwap_baseline). Always "fills".
                    lo = int(lo_idx[j])
                    hi = lo + tau
                    typ = (high_arr[lo:hi] + low_arr[lo:hi] + close_arr[lo:hi]) / 3.0
                    vv = vol_arr[lo:hi]
                    vsum = float(vv.sum())
                    vwap = float((typ * vv).sum() / vsum) if vsum > 0 else float(typ.mean())
                    vwap_slip.append(_slip(side, vwap, open_j, tick))

        # Fold j-1 into the sorted lists and j's outcome into BOTH counters, for next cycle.
        if not np.isnan(v_prev):
            bisect.insort(sv, float(v_prev))
        if not np.isnan(r_prev):
            bisect.insort(sr, float(r_prev))
        if not np.isnan(d_prev):
            bisect.insort(sd, float(d_prev))
        if cell is not None:
            ell_obs = ell_u[j] if side == "sell" else ell_d[j]
            counts.setdefault(cell, Counter())[ell_obs] += 1
            counts_global[ell_obs] += 1

    return {
        "conditional": {t: _summarize(cond_fill[t], cond_slip[t]) for t in targets},
        "unconditional": {t: _summarize(uncond_fill[t], uncond_slip[t]) for t in targets},
        "persistence": _summarize(pers_fill, pers_slip),
        "random": _summarize(rand_fill, rand_slip),
        "vwap": _summarize(len(vwap_slip), vwap_slip),  # VWAP fill ≡ 100% by construction.
    }


def _assert_matches_v2(df_ohlcv, tau, tick, proper_days, side: Side) -> None:
    """Guard: the conditional rung must equal run_backtest_rolling at ASSERT_TARGET.

    Proves this self-contained loop is the same decision path as Stream C's trusted v2 backtest.
    """
    mine = run_ablation(df_ohlcv, tau, tick, proper_days, side, targets=[ASSERT_TARGET])
    c = mine["conditional"][ASSERT_TARGET]
    theirs = run_backtest_rolling(
        df_ohlcv,
        tau=tau,
        tick=tick,
        proper_days=proper_days,
        side=side,
        fill_rate_target=ASSERT_TARGET,
        half_life=HALF_LIFE,
        M=M,
        N=N,
        K=K,
        j_start=J_START,
    )
    assert c.n_decisions == theirs.n_decisions, (c.n_decisions, theirs.n_decisions)
    assert c.n_filled == theirs.n_filled, (c.n_filled, theirs.n_filled)
    assert abs(c.avg - theirs.avg_slippage_ticks) < 1e-9, (c.avg, theirs.avg_slippage_ticks)


def _find_market_dir(substring: str) -> Path | None:
    data = ROOT / "data"
    if not data.exists():
        return None
    for p in data.iterdir():
        if p.is_dir() and substring.lower() in p.name.lower():
            return p
    return None


def _plot_pareto(label: str, side_results: dict[str, dict]) -> None:
    """Pareto: mean slippage vs achieved fill at τ=TAU_FOR_PARETO. conditional/unconditional
    as curves over TARGETS; persistence/random/VWAP as single points. Curve dominance = edge."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, side in zip(axes, ("sell", "buy"), strict=True):
        res = side_results[side]
        for rung, color, marker in (
            ("conditional", "tab:red", "o"),
            ("unconditional", "tab:blue", "s"),
        ):
            xs = [res[rung][t].fill_rate for t in TARGETS]
            ys = [res[rung][t].avg for t in TARGETS]
            ax.plot(xs, ys, marker=marker, color=color, label=rung)
            for t in TARGETS:
                ax.annotate(
                    f"{t:.1f}",
                    (res[rung][t].fill_rate, res[rung][t].avg),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=7,
                    color=color,
                )
        for rung, color, marker in (
            ("persistence", "tab:green", "^"),
            ("random", "tab:gray", "x"),
            ("vwap", "tab:purple", "*"),
        ):
            ax.scatter(
                [res[rung].fill_rate],
                [res[rung].avg],
                color=color,
                marker=marker,
                s=80,
                label=rung,
                zorder=5,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_title(f"{side}")
        ax.set_xlabel("achieved fill rate")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean slippage vs open (ticks; + = beats open)")
    fig.suptitle(f"{label} — ablation Pareto (τ={TAU_FOR_PARETO}); labels = fill_rate_target")
    fig.tight_layout()
    path = REPORTS / f"ablation_pareto_{label}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def _plot_tau(label: str, by_tau: dict[int, dict[str, dict]]) -> None:
    """Mean slippage vs τ at TARGET_FOR_TAU, one line per rung — does any edge survive τ?"""
    taus = sorted(by_tau.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, side in zip(axes, ("sell", "buy"), strict=True):
        for rung, color, marker in (
            ("conditional", "tab:red", "o"),
            ("unconditional", "tab:blue", "s"),
        ):
            ys = [by_tau[tau][side][rung][TARGET_FOR_TAU].avg for tau in taus]
            ax.plot(taus, ys, marker=marker, color=color, label=rung)
        for rung, color, marker in (
            ("persistence", "tab:green", "^"),
            ("random", "tab:gray", "x"),
            ("vwap", "tab:purple", "*"),
        ):
            ys = [by_tau[tau][side][rung].avg for tau in taus]
            ax.plot(taus, ys, marker=marker, color=color, linestyle=":", label=rung)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_title(f"{side}")
        ax.set_xlabel("τ (minutes)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"mean slippage vs open (ticks) @ target={TARGET_FOR_TAU}")
    fig.suptitle(f"{label} — edge vs holding period τ")
    fig.tight_layout()
    path = REPORTS / f"ablation_tau_{label}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def _print_rows(label: str, tau: int, side: str, res: dict) -> None:
    """One block of table rows for a (market, τ, side). Shared n confirms identical sets."""

    def line(rung: str, r: RungResult, tgt: str) -> None:
        print(
            f"{label:9} {side:4} {tau:>3} {tgt:>6} {rung:12} {r.n_decisions:>6} "
            f"{r.fill_rate:>5.1%} {r.avg:>+8.3f} {r.median:>+8.3f} {r.p05:>+9.2f}"
        )

    for tgt in TARGETS:
        line("conditional", res["conditional"][tgt], f"{tgt:.2f}")
        line("unconditional", res["unconditional"][tgt], f"{tgt:.2f}")
    for rung in FLAT_RUNGS:
        line(rung, res[rung], "  -  ")


def sweep_one_market(substring: str, label: str) -> None:
    market_dir = _find_market_dir(substring)
    if market_dir is None:
        print(f"[{label}] data dir not found")
        return
    df = load_market_indexed(market_dir)
    if df.empty:
        print(f"[{label}] no usable bars after roll + liquidity filter")
        return

    first_stem = df["contract"].unique().tolist()[0]
    df_ohlcv = df[["open", "high", "low", "close", "volume"]]
    _, inferred_tick, proper_days, n_green, n_total = _compute_stats(df_ohlcv)
    tick = resolve_tick(first_stem, inferred_tick)
    print(f"\n=== {label} (tick={tick:g}, active days {n_green}/{n_total}) ===")
    print(
        f"{'market':9} {'side':4} {'tau':>3} {'target':>6} {'rung':12} {'n':>6} "
        f"{'fill':>6} {'avg':>8} {'median':>8} {'p05(tail)':>9}"
    )
    print("-" * 78)

    by_tau: dict[int, dict[str, dict]] = {}
    for tau in TAUS:
        by_tau[tau] = {}
        for side in ("sell", "buy"):
            _assert_matches_v2(df_ohlcv, tau, tick, proper_days, side)  # conditional == v2
            res = run_ablation(df_ohlcv, tau, tick, proper_days, side)
            by_tau[tau][side] = res
            _print_rows(label, tau, side, res)

    _plot_pareto(label, by_tau[TAU_FOR_PARETO])
    _plot_tau(label, by_tau)


def main() -> None:
    # Console may be cp1252 (Windows) — force UTF-8 so the banner/figures notes don't crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(
        "Ablation ladder — is the 27-cell regime conditioning real?\n"
        "All rungs scored on the SAME decision windows (gated on the conditional cell).\n"
        "p05 = 5th-percentile slippage (the adverse tail). + = beats a market order at the open.\n"
        f"conditional reproduces run_backtest_rolling at target={ASSERT_TARGET} (asserted)."
    )
    for substring, label in MARKETS:
        sweep_one_market(substring, label)
    print(f"\nFigures in {REPORTS.relative_to(ROOT)}/  (ablation_pareto_*, ablation_tau_*)")


if __name__ == "__main__":
    main()
