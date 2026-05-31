"""Agent execution-value across ALL provided assets (genericity check).

Auto-discovers every market with an ``AIAgent_*.csv`` and, on each, evaluates the
regime-conditioned execution of the agent's real trades plus the winning
*regime-limit + chase-cap* policy. Prints one consolidated table, saves it to
``reports/agent_all_assets.csv`` and a cross-market figure to
``reports/figures/agent_all_assets.png``.

Shortfall is implementation shortfall in ticks vs the OHLC window open (basis-immune;
+ve = beat market-on-decision). The MEDIAN is the honest headline; the mean lives or
dies on the chase tail — which the chase-cap truncates.

Run:
    python scripts/run_agent_all_assets.py
    python scripts/run_agent_all_assets.py --tau 5 --fill-target 0.6 --min-decisions 30
"""

from __future__ import annotations

import argparse
import csv
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
    load_agent_series,
    load_market_for_agent,
)
from order_mgmt.agent.metrics import evaluate_agent_execution  # noqa: E402

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CAPS = (4, 6, 8, 10, 12, 16)


def _stats(arr) -> tuple[float, float, float]:
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return 0.0, 0.0, 0.0
    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 5))


def discover_markets() -> list[Path]:
    data = ROOT / "data"
    return sorted(p for p in data.iterdir() if p.is_dir() and find_agent_csv(p) is not None)


def eval_market(market_dir: Path, *, tau: int, half_life: int, M: int, N: int, K: int,
                j_start: int, fill_target: float) -> dict | None:
    agent = load_agent_series(find_agent_csv(market_dir), market=market_dir.name)
    df_ohlcv, tick, proper_days, _stem = load_market_for_agent(market_dir)
    if df_ohlcv.empty:
        return None
    kw = dict(tau=tau, half_life=half_life, M=M, N=N, K=K,
              fill_rate_target=fill_target, j_start=j_start)
    res = evaluate_agent_execution(agent, df_ohlcv, tick, proper_days, **kw)
    if res.n_decisions == 0:
        return None
    tail = evaluate_tail_strategies(agent, df_ohlcv, tick, proper_days, caps=CAPS, **kw)

    reg_mean, reg_med, reg_p5 = _stats(tail.get("regime", {}).get("shortfall", []))
    cap_means = {c: _stats(tail[f"cap{c}"]["shortfall"])[0] for c in CAPS
                 if tail[f"cap{c}"]["shortfall"].size}
    best_cap = max(cap_means, key=cap_means.get) if cap_means else None
    cap_mean = cap_med = cap_p5 = float("nan")
    if best_cap is not None:
        cap_mean, cap_med, cap_p5 = _stats(tail[f"cap{best_cap}"]["shortfall"])

    return {
        "market": market_dir.name, "tick": tick, "n": res.n_decisions,
        "fill": res.fill_rate, "reg_mean": reg_mean, "reg_med": reg_med, "reg_p5": reg_p5,
        "best_cap": best_cap, "cap_mean": cap_mean, "cap_med": cap_med, "cap_p5": cap_p5,
        "v_market": res.value_add_vs_market_ticks, "v_vwap": res.value_add_vs_vwap_ticks,
    }


def save_csv(rows: list[dict]) -> Path:
    path = REPORTS / "agent_all_assets.csv"
    cols = ["market", "tick", "n", "fill", "reg_mean", "reg_med", "reg_p5",
            "best_cap", "cap_mean", "cap_med", "cap_p5", "v_market", "v_vwap"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    return path


def save_figure(rows: list[dict], min_decisions: int) -> Path:
    """Two panels: mean (uncapped regime vs best chase-cap) and p5 tail, per market."""
    rows = sorted(rows, key=lambda r: r["market"])
    labels = [r["market"][:10] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(x - w / 2, [r["reg_mean"] for r in rows], w, label="regime (uncapped)", color="#8a93a6")
    ax1.bar(x + w / 2, [r["cap_mean"] for r in rows], w, label="regime + best chase-cap", color="#4CAF50")
    ax1.axhline(0, color="black", lw=0.7)
    ax1.set_title("Mean shortfall (ticks; +ve beats open)")
    ax1.set_ylabel("ticks")

    ax2.bar(x - w / 2, [r["reg_p5"] for r in rows], w, label="regime (uncapped)", color="#8a93a6")
    ax2.bar(x + w / 2, [r["cap_p5"] for r in rows], w, label="regime + best chase-cap", color="#ED9E3B")
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_title("5th-percentile tail (ticks; less negative = shorter tail)")
    ax2.set_ylabel("ticks")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend(fontsize=8)
        for i, r in enumerate(rows):
            if r["n"] < min_decisions:
                ax.annotate("low N", (i, 0), ha="center", va="bottom", fontsize=7, color="crimson")

    fig.suptitle("Agent execution-value across all assets — chase-cap vs uncapped regime limit")
    plt.tight_layout()
    path = FIGS / "agent_all_assets.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=int, default=5)
    ap.add_argument("--half-life", type=int, default=20)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--j-start", type=int, default=200)
    ap.add_argument("--fill-target", type=float, default=0.6)
    ap.add_argument("--min-decisions", type=int, default=30,
                    help="flag markets with fewer evaluated decisions as low-confidence")
    args = ap.parse_args()

    markets = discover_markets()
    print(f"Discovered {len(markets)} markets with an AIAgent_*.csv: "
          f"{', '.join(p.name.split()[0] for p in markets)}\n")

    rows: list[dict] = []
    for md in markets:
        r = eval_market(md, tau=args.tau, half_life=args.half_life, M=args.M, N=args.N,
                        K=args.K, j_start=args.j_start, fill_target=args.fill_target)
        if r is None:
            print(f"  [{md.name.split()[0]}] no evaluable decisions — skipped")
            continue
        rows.append(r)

    hdr = (f"{'market':<12}{'tick':>7}{'n':>6}{'fill':>6}{'reg_mean':>9}{'reg_med':>8}"
           f"{'cap':>5}{'cap_mean':>9}{'cap_med':>8}{'cap_p5':>8}{'v_vwap':>8}")
    print("\n=== Agent execution-value - all assets "
          f"(tau={args.tau}, m={args.half_life}, M=N=K={args.M}, fill={args.fill_target}) ===")
    print(hdr)
    for r in rows:
        flag = "  *low-N" if r["n"] < args.min_decisions else ""
        print(f"{r['market'][:11]:<12}{r['tick']:>7g}{r['n']:>6}{r['fill']:>6.2f}"
              f"{r['reg_mean']:>+9.2f}{r['reg_med']:>+8.1f}{r['best_cap'] or 0:>5}"
              f"{r['cap_mean']:>+9.2f}{r['cap_med']:>+8.1f}{r['cap_p5']:>+8.1f}{r['v_vwap']:>+8.2f}{flag}")

    if rows:
        csv_path = save_csv(rows)
        fig_path = save_figure(rows, args.min_decisions)
        print(f"\nsaved: {csv_path.relative_to(ROOT)}")
        print(f"saved: {fig_path.relative_to(ROOT)}")
        n_low = sum(1 for r in rows if r["n"] < args.min_decisions)
        if n_low:
            print(f"\nNote: {n_low} market(s) below {args.min_decisions} decisions - "
                  "their mean/tail are not statistically reliable (data-coverage limited).")


if __name__ == "__main__":
    main()
