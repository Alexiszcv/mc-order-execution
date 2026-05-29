"""Fill-rate sweep: trace the slippage / fill-rate trade-off of the conditional strategy.

For each `fill_rate_target` in a configurable grid, runs the trustworthy no-lookahead
backtest (`run_backtest_rolling`, v2) per side on several markets and records the
achieved fill rate, mean / median slippage, and the 5th-percentile tail. Plots the
mean-slippage-vs-fill-rate Pareto curve per market and prints a table.

This surfaces *where* on the curve a target lands — it does NOT pick a winner
(CLAUDE.md: the parameter choice is the user's). 0.6 is only the current default, not a
recommendation.

Run: `python scripts/sweep_fill_rate.py` from the repo root.
Outputs `reports/figures/pareto_fill_rate_<market>.png` and a summary table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from order_mgmt.backtest import run_backtest_rolling  # noqa: E402
from order_mgmt.pipeline import load_market_indexed  # noqa: E402
from order_mgmt.ticks import resolve_tick  # noqa: E402
from plot_volume import _compute_stats  # noqa: E402

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

# No RNG in this deterministic backtest, but seed for reproducibility hygiene (CLAUDE.md).
SEED = 0

TAU = 5
HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200
TARGETS = [0.4, 0.5, 0.6, 0.7, 0.8]

# (folder-name substring, display label) — extended to 5 markets to prove genericity.
MARKETS = [
    ("Gold", "Gold"),
    ("Nasdaq", "Nasdaq"),
    ("Bunds", "Bunds"),
    ("EuroStoxx", "EuroStoxx"),
    ("GBP", "GBP"),
]

SIDE_STYLE = {"buy": ("tab:blue", "o"), "sell": ("tab:red", "s")}


def _find_market_dir(substring: str) -> Path | None:
    data = ROOT / "data"
    if not data.exists():
        return None
    for p in data.iterdir():
        if p.is_dir() and substring.lower() in p.name.lower():
            return p
    return None


def sweep_one_market(substring: str, label: str) -> dict | None:
    market_dir = _find_market_dir(substring)
    if market_dir is None:
        print(f"[{label}] data dir not found")
        return None

    df = load_market_indexed(market_dir)
    if df.empty:
        print(f"[{label}] no usable bars after roll + liquidity filter")
        return None

    first_stem = df["contract"].unique().tolist()[0]
    df_ohlcv = df[["open", "high", "low", "close", "volume"]]
    _, inferred_tick, proper_days, n_green, n_total = _compute_stats(df_ohlcv)
    tick = resolve_tick(first_stem, inferred_tick)
    print(f"\n=== {label} (tick={tick:g}, active days {n_green}/{n_total}) ===")

    # rows[side] = list of dicts, one per target.
    rows: dict[str, list[dict]] = {"buy": [], "sell": []}
    for side in ("buy", "sell"):
        for target in TARGETS:
            r = run_backtest_rolling(
                df_ohlcv,
                tau=TAU, tick=tick, proper_days=proper_days, side=side,
                fill_rate_target=target, half_life=HALF_LIFE,
                M=M, N=N, K=K, j_start=J_START,
            )
            slips = r.slippage_ticks
            p05 = float(np.percentile(slips, 5)) if slips else 0.0
            rows[side].append({
                "target": target,
                "n": r.n_decisions,
                "fill": r.fill_rate,
                "avg": r.avg_slippage_ticks,
                "median": r.median_slippage_ticks,
                "p05": p05,
            })
            print(
                f"  [{side}] target={target:.2f} n={r.n_decisions:>6} "
                f"fill={r.fill_rate:.1%} avg={r.avg_slippage_ticks:+.3f}t "
                f"med={r.median_slippage_ticks:+.3f}t p05={p05:+.2f}t"
            )

    _plot_pareto(label, rows)
    return rows


def _plot_pareto(label: str, rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for side, recs in rows.items():
        if not recs:
            continue
        color, marker = SIDE_STYLE[side]
        xs = [r["fill"] for r in recs]
        ys = [r["avg"] for r in recs]
        ax.plot(xs, ys, marker=marker, color=color, label=f"{side} (mean)")
        for r in recs:
            ax.annotate(
                f"{r['target']:.1f}", (r["fill"], r["avg"]),
                textcoords="offset points", xytext=(4, 4), fontsize=8, color=color,
            )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
    ax.set_title(f"{label} — slippage vs achieved fill rate\n(labels = fill_rate_target)")
    ax.set_xlabel("achieved fill rate")
    ax.set_ylabel("mean slippage vs open (ticks; + = strategy beats open)")
    ax.legend()
    plt.tight_layout()
    path = REPORTS / f"pareto_fill_rate_{label}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def _print_summary(all_results: dict[str, dict]) -> None:
    print("\n=== Summary: mean/median slippage & tail by target (v2, no-lookahead) ===")
    hdr = f"{'market':10} {'side':4} {'target':>6} {'n':>6} {'fill':>6} {'avg':>8} {'median':>8} {'p05':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, rows in all_results.items():
        for side in ("buy", "sell"):
            for r in rows[side]:
                print(
                    f"{label:10} {side:4} {r['target']:>6.2f} {r['n']:>6} "
                    f"{r['fill']:>5.1%} {r['avg']:>+8.3f} {r['median']:>+8.3f} {r['p05']:>+8.2f}"
                )


def main() -> None:
    np.random.seed(SEED)
    all_results: dict[str, dict] = {}
    for substring, label in MARKETS:
        res = sweep_one_market(substring, label)
        if res is not None:
            all_results[label] = res
    _print_summary(all_results)
    print(f"\nMarkets swept: {len(all_results)} (need >=2). Figures in {REPORTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
