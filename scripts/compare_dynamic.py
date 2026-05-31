"""Compare dynamic (sequential) execution against single-shot and slicing, same orders.

Five schemes on identical agent orders / regime / ℓ* (all within the τ window):
  single   — one limit for the whole order (baseline)
  blend    — half at the open, half on a limit
  cutoff   — limit, then market the remainder mid-window if unfilled
  adaptive — re-post each minute; shrink ℓ* as the deadline nears (rule)
  dp       — re-post each minute; ℓ*(remaining, regime) from backward induction (optimal)

Deciding lens: the median (typical price improvement) vs the 5th-pct tail (worst case).
The dynamic policies should push the median/tail frontier out beyond the static schemes.

Run: python scripts/compare_dynamic.py --market Gold --market Nasdaq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from order_mgmt.agent.dynamic import evaluate_agent_dynamic  # noqa: E402
from order_mgmt.agent.loader import (  # noqa: E402
    find_agent_csv,
    find_market_dir,
    load_agent_series,
    load_market_for_agent,
)

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

TAU = 5
FILL_RATE_TARGET = 0.6
HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200
ORDER = ["single", "blend", "cutoff", "adaptive", "dp"]
COLORS = {
    "single": "0.4", "blend": "tab:green", "cutoff": "tab:red",
    "adaptive": "tab:blue", "dp": "tab:purple",
}
LABELS = {
    "single": "single-shot", "blend": "blend", "cutoff": "cutoff",
    "adaptive": "adaptive (rule)", "dp": "DP (optimal)",
}


def compare_market(substring: str) -> None:
    market_dir = find_market_dir(ROOT / "data", substring)
    if market_dir is None or find_agent_csv(market_dir) is None:
        print(f"[{substring}] market dir or AIAgent csv not found")
        return
    df_ohlcv, tick, proper_days, _ = load_market_for_agent(market_dir)
    agent = load_agent_series(find_agent_csv(market_dir), market=market_dir.name)

    res = evaluate_agent_dynamic(
        agent, df_ohlcv, tick=tick, proper_days=proper_days, tau=TAU, half_life=HALF_LIFE,
        M=M, N=N, K=K, fill_rate_target=FILL_RATE_TARGET, j_start=J_START,
    )

    print(f"\n=== {market_dir.name} (tick={tick:g}, tau={TAU}, target={FILL_RATE_TARGET}) ===")
    print(f"  {'scheme':16} {'n':>6} {'mean':>8} {'median':>8} {'p5 tail':>9} {'p1 tail':>9}")
    stats: dict[str, dict] = {}
    for name in ORDER:
        s = res[name]["shortfall"]
        if s.size == 0:
            continue
        stats[name] = {
            "mean": float(s.mean()), "median": float(np.median(s)),
            "p5": float(np.percentile(s, 5)), "p1": float(np.percentile(s, 1)), "vals": s,
        }
        st = stats[name]
        print(
            f"  {LABELS[name]:16} {s.size:>6} {st['mean']:>+8.2f} {st['median']:>+8.2f} "
            f"{st['p5']:>+9.1f} {st['p1']:>+9.1f}"
        )
    _plot(market_dir.name, stats)


def _plot(label: str, stats: dict) -> None:
    if not stats:
        return
    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(15, 5.5))
    lo, hi = -40, 25
    for name in ORDER:
        if name not in stats:
            continue
        vals = np.clip(stats[name]["vals"], lo, hi)
        ax_h.hist(vals, bins=60, range=(lo, hi), histtype="step", linewidth=1.6,
                  color=COLORS[name], label=f"{LABELS[name]} (med {stats[name]['median']:+.0f}t)")
    ax_h.axvline(0, color="black", linestyle="--", linewidth=0.6)
    ax_h.set_title(f"{label} — shortfall distribution by scheme (clipped [{lo},{hi}] ticks)")
    ax_h.set_xlabel("shortfall (ticks; +ve = beat arrival)")
    ax_h.set_ylabel("count")
    ax_h.legend(fontsize=8)

    names = [n for n in ORDER if n in stats]
    x = np.arange(len(names))
    width = 0.38
    ax_b.bar(x - width / 2, [stats[n]["median"] for n in names], width, label="median (headline)", color="seagreen")
    ax_b.bar(x + width / 2, [stats[n]["p5"] for n in names], width, label="5th-pct (tail)", color="indianred")
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([LABELS[n] for n in names], fontsize=8, rotation=15)
    ax_b.set_ylabel("ticks")
    ax_b.set_title(f"{label} — median vs tail (higher/less-negative = better)")
    ax_b.legend(fontsize=8)

    fig.tight_layout()
    path = REPORTS / f"agent_dynamic_compare_{label.split()[0]}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", action="append", default=[], help="market substring (repeatable)")
    args = ap.parse_args()
    for substring in args.market or ["Gold", "Nasdaq"]:
        compare_market(substring)


if __name__ == "__main__":
    main()
