"""Is the regime-conditioned execution actually good? Compare it to dumb baselines.

market (all-in) | random offset | global ℓ* (no regime) | regime ℓ* (ours) | DP (complex).
The decisive controls: regime must beat `random` (skill > guessing) and `global`
(regime conditioning > one pooled limit); DP shows what the complex policy adds.

Run: python scripts/compare_benchmarks.py --market Gold --market Nasdaq
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

from order_mgmt.agent.benchmarks import NAMES, evaluate_agent_benchmarks  # noqa: E402
from order_mgmt.agent.loader import (  # noqa: E402
    find_agent_csv,
    find_market_dir,
    load_agent_series,
    load_market_for_agent,
)

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

TAU, FILL_RATE_TARGET, HALF_LIFE, J_START = 5, 0.6, 20, 200
M, N, K = 3, 3, 3
LABELS = {
    "market": "market (all-in)", "random": "random offset", "global": "global L* (no regime)",
    "regime": "regime L* (ours)", "dp": "DP (complex)",
}
COLORS = {"market": "0.5", "random": "tab:orange", "global": "tab:olive", "regime": "tab:blue", "dp": "tab:purple"}


def compare_market(substring: str) -> None:
    market_dir = find_market_dir(ROOT / "data", substring)
    if market_dir is None or find_agent_csv(market_dir) is None:
        print(f"[{substring}] not found")
        return
    df_ohlcv, tick, proper_days, _ = load_market_for_agent(market_dir)
    agent = load_agent_series(find_agent_csv(market_dir), market=market_dir.name)
    res = evaluate_agent_benchmarks(
        agent, df_ohlcv, tick=tick, proper_days=proper_days, tau=TAU, half_life=HALF_LIFE,
        M=M, N=N, K=K, fill_rate_target=FILL_RATE_TARGET, j_start=J_START, seed=0,
    )

    print(f"\n=== {market_dir.name} (tick={tick:g}, tau={TAU}, target={FILL_RATE_TARGET}) ===")
    print(f"  {'strategy':22} {'n':>6} {'mean':>8} {'median':>8} {'p5 tail':>9}")
    stats = {}
    for nm in NAMES:
        s = res[nm]["shortfall"]
        if s.size == 0:
            continue
        stats[nm] = {"mean": float(s.mean()), "median": float(np.median(s)),
                     "p5": float(np.percentile(s, 5)), "vals": s}
        st = stats[nm]
        print(f"  {LABELS[nm]:22} {s.size:>6} {st['mean']:>+8.2f} {st['median']:>+8.2f} {st['p5']:>+9.1f}")
    _plot(market_dir.name, stats)


def _plot(label: str, stats: dict) -> None:
    if not stats:
        return
    names = [n for n in NAMES if n in stats]
    fig, (ax_b, ax_h) = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(names))
    width = 0.38
    ax_b.bar(x - width / 2, [stats[n]["median"] for n in names], width, label="median", color="seagreen")
    ax_b.bar(x + width / 2, [stats[n]["p5"] for n in names], width, label="p5 tail", color="indianred")
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([LABELS[n] for n in names], fontsize=7, rotation=20, ha="right")
    ax_b.set_ylabel("ticks")
    ax_b.set_title(f"{label} — median & tail by strategy (higher/less-neg = better)")
    ax_b.legend(fontsize=8)

    lo, hi = -40, 25
    for n in names:
        ax_h.hist(np.clip(stats[n]["vals"], lo, hi), bins=60, range=(lo, hi), histtype="step",
                  linewidth=1.5, color=COLORS[n], label=f"{LABELS[n]} (med {stats[n]['median']:+.0f})")
    ax_h.axvline(0, color="black", linestyle="--", linewidth=0.6)
    ax_h.set_title(f"{label} — shortfall distribution (clipped [{lo},{hi}])")
    ax_h.set_xlabel("shortfall (ticks; +ve = beat arrival)")
    ax_h.legend(fontsize=7)

    fig.tight_layout()
    path = REPORTS / f"agent_benchmarks_{label.split()[0]}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", action="append", default=[])
    args = ap.parse_args()
    for substring in args.market or ["Gold", "Nasdaq"]:
        compare_market(substring)


if __name__ == "__main__":
    main()
