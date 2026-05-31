"""Tail reduction & the mean: can a chase-cap beat plain market execution on the MEAN?

The mean is negative because of the chase tail. A symmetric tighten (smaller ℓ*) shrinks
wins and losses alike → converges to market. A chase-CAP is asymmetric: keep the limit
upside, stop out at `cap` ticks of adverse move. Sweeping `cap` traces the mean/tail
frontier and shows whether some cap makes the mean POSITIVE (beating market).

Run: python scripts/compare_tail_reduction.py --market Gold --market Nasdaq
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

from order_mgmt.agent.benchmarks import evaluate_tail_strategies  # noqa: E402
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
CAPS = (4, 6, 8, 10, 12, 16, 24)


def _agg(s: np.ndarray) -> dict:
    return {"mean": float(s.mean()), "median": float(np.median(s)), "p5": float(np.percentile(s, 5))}


def compare_market(substring: str) -> None:
    market_dir = find_market_dir(ROOT / "data", substring)
    if market_dir is None or find_agent_csv(market_dir) is None:
        print(f"[{substring}] not found")
        return
    df_ohlcv, tick, proper_days, _ = load_market_for_agent(market_dir)
    agent = load_agent_series(find_agent_csv(market_dir), market=market_dir.name)
    res = evaluate_tail_strategies(
        agent, df_ohlcv, tick=tick, proper_days=proper_days, tau=TAU, half_life=HALF_LIFE,
        M=M, N=N, K=K, fill_rate_target=FILL_RATE_TARGET, j_start=J_START, caps=CAPS,
    )
    stats = {nm: _agg(res[nm]["shortfall"]) for nm in res if res[nm]["shortfall"].size}

    print(f"\n=== {market_dir.name} (tick={tick:g}, tau={TAU}) ===")
    print(f"  {'strategy':14} {'mean':>8} {'median':>8} {'p5 tail':>9}")
    for nm in ("market", "regime", "dp", *[f"cap{c}" for c in CAPS]):
        if nm in stats:
            st = stats[nm]
            print(f"  {nm:14} {st['mean']:>+8.2f} {st['median']:>+8.2f} {st['p5']:>+9.1f}")
    _plot(market_dir.name, stats)


def _plot(label: str, stats: dict) -> None:
    cap_means = [stats[f"cap{c}"]["mean"] for c in CAPS if f"cap{c}" in stats]
    cap_p5 = [stats[f"cap{c}"]["p5"] for c in CAPS if f"cap{c}" in stats]
    xs = [c for c in CAPS if f"cap{c}" in stats]
    fig, (axm, axt) = plt.subplots(1, 2, figsize=(14, 5))

    axm.plot(xs, cap_means, "o-", color="tab:blue", label="capped strategy")
    axm.axhline(stats["market"]["mean"], color="0.5", ls="--", label=f"market ({stats['market']['mean']:+.2f})")
    axm.axhline(stats["regime"]["mean"], color="tab:red", ls=":", label=f"regime ({stats['regime']['mean']:+.2f})")
    axm.axhline(0, color="black", lw=0.5)
    axm.set_title(f"{label} — MEAN shortfall vs chase-cap (higher = better)")
    axm.set_xlabel("cap (ticks of adverse move before stopping)")
    axm.set_ylabel("mean shortfall (ticks)")
    axm.legend(fontsize=8)

    axt.plot(xs, cap_p5, "o-", color="tab:blue", label="capped strategy")
    axt.axhline(stats["market"]["p5"], color="0.5", ls="--", label=f"market ({stats['market']['p5']:+.1f})")
    axt.axhline(stats["regime"]["p5"], color="tab:red", ls=":", label=f"regime ({stats['regime']['p5']:+.1f})")
    axt.set_title(f"{label} — TAIL (5th-pct) vs chase-cap (higher = better)")
    axt.set_xlabel("cap (ticks)")
    axt.set_ylabel("p5 shortfall (ticks)")
    axt.legend(fontsize=8)

    fig.tight_layout()
    path = REPORTS / f"agent_tailcap_{label.split()[0]}.png"
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
