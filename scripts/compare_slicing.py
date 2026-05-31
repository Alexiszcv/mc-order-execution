"""Compare order-slicing schemes against single-shot execution, on the same agent orders.

The agent supplies side+timing; our model supplies ℓ*. We then fill each parent order four
ways (all within the τ window): single-shot, time-slice (K=3), passive/aggressive blend
(f=0.5), and limit-then-market cutoff (0.5). Because the regime, ℓ* and window bars are
shared, the ONLY difference is how the order is cut.

The deciding metric is the TAIL: slicing should trade a little median improvement for a much
thinner unfilled-chase tail (5th-percentile shortfall). If it tames the tail without killing
the median, slice; if it just dilutes the edge, don't.

Run: python scripts/compare_slicing.py --market Gold --market Nasdaq
Outputs reports/figures/agent_slicing_compare_<market>.png and prints a table.
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

from order_mgmt.agent.loader import (  # noqa: E402
    find_agent_csv,
    find_market_dir,
    load_agent_series,
    load_market_for_agent,
)
from order_mgmt.agent.metrics import evaluate_agent_schemes  # noqa: E402
from order_mgmt.agent.slicing import SCHEMES  # noqa: E402

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

TAU = 5  # match the agent's 5-min decision cadence
FILL_RATE_TARGET = 0.6
HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200
ORDER = ["single", "time_slice", "blend", "cutoff"]
COLORS = {"single": "0.4", "time_slice": "tab:blue", "blend": "tab:green", "cutoff": "tab:red"}
LABELS = {
    "single": "single-shot",
    "time_slice": "time-slice K=3",
    "blend": "blend f=0.5",
    "cutoff": "cutoff 0.5",
}


def compare_market(substring: str) -> None:
    market_dir = find_market_dir(ROOT / "data", substring)
    if market_dir is None or find_agent_csv(market_dir) is None:
        print(f"[{substring}] market dir or AIAgent csv not found")
        return
    df_ohlcv, tick, proper_days, _ = load_market_for_agent(market_dir)
    agent = load_agent_series(find_agent_csv(market_dir), market=market_dir.name)

    res = evaluate_agent_schemes(
        agent, df_ohlcv, tick=tick, proper_days=proper_days, tau=TAU, half_life=HALF_LIFE,
        M=M, N=N, K=K, fill_rate_target=FILL_RATE_TARGET, j_start=J_START, schemes=SCHEMES,
    )

    print(f"\n=== {market_dir.name} (tick={tick:g}, tau={TAU}, target={FILL_RATE_TARGET}) ===")
    print(f"  {'scheme':16} {'n':>6} {'fill':>7} {'mean':>8} {'median':>8} {'p5 tail':>9} {'p1 tail':>9}")
    stats: dict[str, dict] = {}
    for name in ORDER:
        s = res[name]["shortfall"]
        frac = res[name]["fill_frac"]
        if s.size == 0:
            continue
        stats[name] = {
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "p5": float(np.percentile(s, 5)),
            "p1": float(np.percentile(s, 1)),
            "fill": float(frac.mean()),
            "vals": s,
        }
        st = stats[name]
        print(
            f"  {LABELS[name]:16} {s.size:>6} {st['fill']:>6.1%} "
            f"{st['mean']:>+8.2f} {st['median']:>+8.2f} {st['p5']:>+9.1f} {st['p1']:>+9.1f}"
        )

    _plot(market_dir.name, stats)


def _plot(label: str, stats: dict) -> None:
    if not stats:
        return
    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel 1: shortfall distributions (clipped for display so the tail is visible).
    lo, hi = -40, 25
    for name in ORDER:
        if name not in stats:
            continue
        vals = np.clip(stats[name]["vals"], lo, hi)
        ax_h.hist(vals, bins=60, range=(lo, hi), histtype="step", linewidth=1.6,
                  color=COLORS[name], label=f"{LABELS[name]} (med {stats[name]['median']:+.0f}t)")
    ax_h.axvline(0, color="black", linestyle="--", linewidth=0.6)
    ax_h.set_title(f"{label} — shortfall distribution by scheme (clipped to [{lo},{hi}] ticks)")
    ax_h.set_xlabel("shortfall (ticks; +ve = beat arrival)")
    ax_h.set_ylabel("count")
    ax_h.legend(fontsize=8)

    # Panel 2: median (headline) vs p5 tail (worst-case) per scheme.
    names = [n for n in ORDER if n in stats]
    x = np.arange(len(names))
    width = 0.38
    ax_b.bar(x - width / 2, [stats[n]["median"] for n in names], width, label="median (headline)", color="seagreen")
    ax_b.bar(x + width / 2, [stats[n]["p5"] for n in names], width, label="5th-pct (tail)", color="indianred")
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([LABELS[n] for n in names], fontsize=8)
    ax_b.set_ylabel("ticks")
    ax_b.set_title(f"{label} — median vs tail (higher/less-negative = better)")
    ax_b.legend(fontsize=8)

    fig.tight_layout()
    path = REPORTS / f"agent_slicing_compare_{label.split()[0]}.png"
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
