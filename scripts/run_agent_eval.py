"""End-to-end AIAgent execution-value evaluation across markets (+ synthetic).

Computes, for each market, how the regime-conditioned limit-order execution performs
on the AI agent's actual trades vs naive baselines — measured as implementation
shortfall in ticks (NOT market-impact slippage; there is no order-book data). The
MEDIAN is the honest headline: the chase tail drags the mean toward zero.

Run:
    python scripts/run_agent_eval.py --market Gold --market Nasdaq --synthetic --seed 0

Outputs a per-market shortfall histogram (real vs synthetic) to reports/figures/
and prints a summary table.
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
from order_mgmt.agent.metrics import AgentEvalResult, evaluate_agent_execution  # noqa: E402
from order_mgmt.agent.synthetic import synth_agent_series  # noqa: E402

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

# Defaults mirror scripts/run_v1.py so results are comparable across the project.
TAU = 5
HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200
FILL_RATE_TARGET = 0.6


def _eval_kwargs(tick: float, proper_days: list) -> dict:
    return dict(
        tick=tick,
        proper_days=proper_days,
        tau=TAU,
        half_life=HALF_LIFE,
        M=M,
        N=N,
        K=K,
        fill_rate_target=FILL_RATE_TARGET,
        j_start=J_START,
    )


def _print_result(label: str, r: AgentEvalResult) -> None:
    print(
        f"  [{label}] n={r.n_decisions} fill={r.fill_rate:.1%} "
        f"shortfall mean={r.mean_shortfall_ticks:+.2f}t median={r.median_shortfall_ticks:+.2f}t "
        f"| vs market(mean)={r.value_add_vs_market_ticks:+.2f}t "
        f"vs vwap(mean)={r.value_add_vs_vwap_ticks:+.2f}t "
        f"| unfilled-tail={r.unfilled_tail_cost_ticks:+.2f}t"
    )


def run_one_market(substring: str, *, with_synthetic: bool, seed: int) -> dict | None:
    market_dir = find_market_dir(ROOT / "data", substring)
    if market_dir is None:
        print(f"[{substring}] data dir not found")
        return None

    df_ohlcv, tick, proper_days, first_stem = load_market_for_agent(market_dir)
    if df_ohlcv.empty:
        print(f"[{substring}] no usable bars after roll + liquidity filter")
        return None
    print(f"\n=== {market_dir.name} (contract {first_stem}, tick={tick:g}) ===")

    out: dict[str, AgentEvalResult] = {}

    agent_csv = find_agent_csv(market_dir)
    if agent_csv is None:
        print(f"  no AIAgent_*.csv in {market_dir.name}")
    else:
        agent = load_agent_series(agent_csv, market=market_dir.name)
        real = evaluate_agent_execution(agent, df_ohlcv, **_eval_kwargs(tick, proper_days))
        _print_result("real", real)
        out["real"] = real

    if with_synthetic:
        synth = synth_agent_series(df_ohlcv, seed=seed, n_decisions=500)
        synth_res = evaluate_agent_execution(synth, df_ohlcv, **_eval_kwargs(tick, proper_days))
        _print_result("synthetic", synth_res)
        out["synthetic"] = synth_res

    _plot_market(market_dir.name, out)
    return out


def _plot_market(label: str, out: dict[str, AgentEvalResult]) -> None:
    series = {k: [f.shortfall_ticks for f in r.fills] for k, r in out.items() if r.fills}
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"real": "steelblue", "synthetic": "darkorange"}
    for name, vals in series.items():
        med = float(np.median(vals))
        ax.hist(vals, bins=50, alpha=0.5, label=f"{name} (median {med:+.1f}t)", color=colors.get(name))
    ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set_title(f"{label} — implementation shortfall vs arrival (ticks; +ve = beat arrival)")
    ax.set_xlabel("shortfall (ticks)")
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    path = REPORTS / f"agent_{label.split()[0]}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", action="append", default=[], help="market name substring (repeatable)")
    ap.add_argument("--synthetic", action="store_true", help="also evaluate a synthetic agent series")
    ap.add_argument("--seed", type=int, default=0, help="seed for the synthetic generator")
    args = ap.parse_args()

    markets = args.market or ["Gold", "Nasdaq"]

    results: dict[str, dict] = {}
    for substring in markets:
        res = run_one_market(substring, with_synthetic=args.synthetic, seed=args.seed)
        if res:
            results[substring] = res

    print("\n=== Summary (implementation shortfall, ticks) ===")
    print(f"{'market':12} {'source':10} {'n':>6} {'fill':>7} {'mean':>8} {'median':>8} {'vs_vwap':>9}")
    for substring, res in results.items():
        for source, r in res.items():
            print(
                f"{substring:12} {source:10} {r.n_decisions:>6} {r.fill_rate:>6.1%} "
                f"{r.mean_shortfall_ticks:>+8.2f} {r.median_shortfall_ticks:>+8.2f} "
                f"{r.value_add_vs_vwap_ticks:>+9.2f}"
            )


if __name__ == "__main__":
    main()
