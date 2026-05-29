"""End-to-end v1 demo: regime-conditioned strategy vs TWAP/VWAP across markets.

Uses the roll-aware multi-contract loader so each market spans its full history
(not just the first contract). Spec tick sizes from `order_mgmt.ticks` override
the heuristic inference when known.

Run: `python scripts/run_v1.py` from the repo root.
Outputs slippage histograms to reports/figures/ and prints a summary table.
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

from order_mgmt.backtest import run_backtest, run_backtest_rolling  # noqa: E402
from order_mgmt.baselines import vwap_baseline  # noqa: E402
from order_mgmt.pipeline import load_market_indexed  # noqa: E402
from order_mgmt.ticks import resolve_tick  # noqa: E402
from plot_volume import _compute_stats  # noqa: E402
from ranges import compute_all_ranges  # noqa: E402
from regime import compute_ewma_series  # noqa: E402

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

TAU = 5
HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200
FILL_RATE_TARGET = 0.6

# (folder-name substring, display label)
MARKETS = [
    ("Gold", "Gold"),
    ("Nasdaq", "Nasdaq"),
]


def _find_market_dir(substring: str) -> Path | None:
    data = ROOT / "data"
    if not data.exists():
        return None
    for p in data.iterdir():
        if p.is_dir() and substring.lower() in p.name.lower():
            return p
    return None


def run_one_market(substring: str, label: str) -> dict | None:
    market_dir = _find_market_dir(substring)
    if market_dir is None:
        print(f"[{label}] data dir not found")
        return None

    df_1min = load_market_indexed(market_dir)
    if df_1min.empty:
        print(f"[{label}] no usable bars after roll + liquidity filter")
        return None

    contracts_used = df_1min["contract"].unique().tolist()
    first_stem = contracts_used[0]
    print(f"\n=== {label} ({len(contracts_used)} contracts rolled: {contracts_used}) ===")

    # Drop the 'contract' column so downstream team code sees only OHLCV
    df_ohlcv = df_1min[["open", "high", "low", "close", "volume"]]

    _, inferred_tick, proper_days, n_green, n_total = _compute_stats(df_ohlcv)
    tick = resolve_tick(first_stem, inferred_tick)
    src = "spec" if tick != inferred_tick else "heuristic"
    print(f"  tick = {tick:g} ({src}; inferred {inferred_tick:g}); active days {n_green}/{n_total}")

    # Compute the range + EWMA passes ONCE per market and reuse them across both
    # backtests (buy/sell) and the VWAP baseline, instead of recomputing each call.
    ranges = compute_all_ranges(df_ohlcv, TAU, tick, proper_days)
    t_list = ranges[0]
    ewma = compute_ewma_series(t_list, ranges[1], ranges[4], HALF_LIFE)

    out: dict[str, dict] = {}
    for side in ("buy", "sell"):
        v1 = run_backtest(
            df_ohlcv,
            tau=TAU, tick=tick, proper_days=proper_days, side=side,
            fill_rate_target=FILL_RATE_TARGET, half_life=HALF_LIFE,
            M=M, N=N, K=K, j_start=J_START, ranges=ranges, ewma=ewma,
        )
        v2 = run_backtest_rolling(
            df_ohlcv,
            tau=TAU, tick=tick, proper_days=proper_days, side=side,
            fill_rate_target=FILL_RATE_TARGET, half_life=HALF_LIFE,
            M=M, N=N, K=K, j_start=J_START, ranges=ranges, ewma=ewma,
        )
        vwap = vwap_baseline(df_ohlcv, t_list[J_START:], tau=TAU, tick=tick, side=side)
        out[side] = {"v1": v1, "v2": v2, "vwap": vwap}

        vwap_avg = float(np.mean(vwap.slippage_ticks)) if vwap.slippage_ticks else 0.0
        print(
            f"  [{side}] v1: n={v1.n_decisions} fill={v1.fill_rate:.1%} "
            f"avg={v1.avg_slippage_ticks:+.2f}t med={v1.median_slippage_ticks:+.2f}t"
        )
        print(
            f"       v2: n={v2.n_decisions} fill={v2.fill_rate:.1%} "
            f"avg={v2.avg_slippage_ticks:+.2f}t med={v2.median_slippage_ticks:+.2f}t  "
            f"(no-lookahead)"
        )
        print(f"       vwap avg={vwap_avg:+.2f}t")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, side in zip(axes, ("buy", "sell"), strict=False):
        v1 = out[side]["v1"]
        v2 = out[side]["v2"]
        v = out[side]["vwap"]
        if v1.slippage_ticks:
            ax.hist(v1.slippage_ticks, bins=40, alpha=0.5, label="Strategy (v1)", color="steelblue")
        if v2.slippage_ticks:
            ax.hist(v2.slippage_ticks, bins=40, alpha=0.5, label="Strategy (v2)", color="seagreen")
        if v.slippage_ticks:
            ax.hist(v.slippage_ticks, bins=40, alpha=0.5, label="VWAP", color="orange")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_title(f"{label} — {side} (slippage vs open)")
        ax.set_xlabel("ticks")
        ax.set_ylabel("count")
        ax.legend()
    plt.tight_layout()
    path = REPORTS / f"slippage_{label}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")
    return out


def main() -> None:
    all_results: dict[str, dict] = {}
    for substring, label in MARKETS:
        res = run_one_market(substring, label)
        if res is not None:
            all_results[label] = res

    print("\n=== Summary ===")
    print(f"{'market':12} {'side':5} {'variant':>8} {'n':>7} {'fill':>7} {'avg':>8} {'median':>8}")
    for label, res in all_results.items():
        for side, s in res.items():
            for variant in ("v1", "v2"):
                r = s[variant]
                print(
                    f"{label:12} {side:5} {variant:>8} {r.n_decisions:>7} {r.fill_rate:>6.1%} "
                    f"{r.avg_slippage_ticks:>+8.2f} {r.median_slippage_ticks:>+8.2f}"
                )
            v = s["vwap"]
            v_avg = float(np.mean(v.slippage_ticks)) if v.slippage_ticks else 0.0
            v_med = float(np.median(v.slippage_ticks)) if v.slippage_ticks else 0.0
            print(
                f"{label:12} {side:5} {'vwap':>8} {len(v.slippage_ticks):>7} {'-':>7} "
                f"{v_avg:>+8.2f} {v_med:>+8.2f}"
            )


if __name__ == "__main__":
    main()
