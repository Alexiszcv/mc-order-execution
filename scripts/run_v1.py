"""End-to-end v1 demo: regime-conditioned strategy vs TWAP/VWAP on two markets.

Run: `python scripts/run_v1.py` from the repo root.
Outputs slippage histograms to reports/figures/.
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

from epdf import _load_1min, compute_all_ranges  # noqa: E402
from plot_volume import _compute_stats  # noqa: E402

from order_mgmt.backtest import run_backtest  # noqa: E402
from order_mgmt.baselines import vwap_baseline  # noqa: E402

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

TAU = 5
HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200
FILL_RATE_TARGET = 0.6

# (market_dir_substring, display_name)
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
    csvs = [p for p in sorted(market_dir.glob("*.csv")) if not p.stem.startswith("AIAgent_")]
    if not csvs:
        print(f"[{label}] no OHLC CSVs in {market_dir}")
        return None
    csv = csvs[0]
    print(f"\n=== {label} ({csv.stem}) ===")

    df_1min = _load_1min(str(csv))
    _, tick, proper_days, n_green, n_total = _compute_stats(df_1min)
    print(f"  tick = {tick:g}; active days {n_green}/{n_total}")

    out: dict[str, dict] = {}
    for side in ("buy", "sell"):
        result = run_backtest(
            df_1min,
            tau=TAU,
            tick=tick,
            proper_days=proper_days,
            side=side,
            fill_rate_target=FILL_RATE_TARGET,
            half_life=HALF_LIFE,
            M=M,
            N=N,
            K=K,
            j_start=J_START,
        )
        t_list, *_ = compute_all_ranges(df_1min, TAU, tick, proper_days)
        vwap = vwap_baseline(df_1min, t_list[J_START:], tau=TAU, tick=tick, side=side)
        out[side] = {"result": result, "vwap": vwap}

        vwap_avg = float(np.mean(vwap.slippage_ticks)) if vwap.slippage_ticks else 0.0
        print(f"  [{side}] n={result.n_decisions}  fill={result.fill_rate:.1%}  "
              f"strategy avg={result.avg_slippage_ticks:+.2f}t  "
              f"median={result.median_slippage_ticks:+.2f}t  vwap avg={vwap_avg:+.2f}t")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, side in zip(axes, ("buy", "sell"), strict=False):
        r = out[side]["result"]
        v = out[side]["vwap"]
        if r.slippage_ticks:
            ax.hist(r.slippage_ticks, bins=40, alpha=0.6, label="Strategy", color="steelblue")
        if v.slippage_ticks:
            ax.hist(v.slippage_ticks, bins=40, alpha=0.6, label="VWAP", color="orange")
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
    print(f"{'market':12} {'side':5} {'n':>6} {'fill':>7} {'strat':>9} {'vwap':>9}")
    for label, res in all_results.items():
        for side, s in res.items():
            r = s["result"]
            v = s["vwap"]
            v_avg = float(np.mean(v.slippage_ticks)) if v.slippage_ticks else 0.0
            print(f"{label:12} {side:5} {r.n_decisions:>6} {r.fill_rate:>6.1%} "
                  f"{r.avg_slippage_ticks:>+8.2f} {v_avg:>+8.2f}")


if __name__ == "__main__":
    main()
