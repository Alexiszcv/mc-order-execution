"""Parameter sweep of agent-execution metrics.

Setup: the AGENT supplies side + timing (and size = |Δposition|); OUR MODEL supplies the
execution policy — the regime-conditioned limit offset ℓ* picked from the ePDF. We sweep
the two knobs that most move execution, fill_rate_target × τ, and plot how each metric
responds, so the trade-offs are visible rather than asserted.

Metrics (per market, per parameter point):
  - fill rate                  — fraction of orders the passive limit fills
  - median shortfall (ticks)   — typical price improvement vs arrival (the honest headline)
  - mean shortfall (ticks)     — dragged by the unfilled-chase tail
  - value-add vs VWAP (ticks)  — mean(our shortfall) − mean(VWAP shortfall)
  - captured improvement       — Σ shortfall·tick·|Δpos|  (size-weighted by the agent's qty)

Run: python scripts/sweep_agent_metrics.py --market Gold [--market Nasdaq]
Outputs reports/figures/agent_sweep_<market>.png and prints the grid.
"""

from __future__ import annotations

import argparse
import math
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
from order_mgmt.agent.metrics import evaluate_agent_execution, size_weighted_shortfall  # noqa: E402

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)

TAUS = [5, 10, 15, 30, 60]  # the spec's holding periods
FILL_TARGETS = [0.4, 0.5, 0.6, 0.7, 0.8]
SIZING_RULES = ("size_agent", "size_confidence", "size_inverse_vol")
RULE_LABEL = {"size_agent": "agent |dpos|", "size_confidence": "confidence", "size_inverse_vol": "inverse-vol"}
HALF_LIFE = 20
M, N, K = 3, 3, 3
J_START = 200


def sweep_market(substring: str) -> tuple[str, dict] | None:
    market_dir = find_market_dir(ROOT / "data", substring)
    if market_dir is None:
        print(f"[{substring}] data dir not found")
        return None
    agent_csv = find_agent_csv(market_dir)
    if agent_csv is None:
        print(f"[{substring}] no AIAgent_*.csv")
        return None

    df_ohlcv, tick, proper_days, _ = load_market_for_agent(market_dir)
    agent = load_agent_series(agent_csv, market=market_dir.name)
    print(f"\n=== {market_dir.name} (tick={tick:g}) sweeping {len(TAUS)}×{len(FILL_TARGETS)} points ===")

    # grid[metric][tau] = list over FILL_TARGETS
    metrics = (
        "fill_rate", "median_shortfall", "mean_shortfall", "value_add_vwap",
        "size_agent", "size_confidence", "size_inverse_vol",
    )
    grid: dict[str, dict[int, list[float]]] = {m: {t: [] for t in TAUS} for m in metrics}

    for tau in TAUS:
        for frt in FILL_TARGETS:
            r = evaluate_agent_execution(
                agent, df_ohlcv, tick=tick, proper_days=proper_days,
                tau=tau, half_life=HALF_LIFE, M=M, N=N, K=K,
                fill_rate_target=frt, j_start=J_START,
            )
            grid["fill_rate"][tau].append(r.fill_rate)
            grid["median_shortfall"][tau].append(r.median_shortfall_ticks)
            grid["mean_shortfall"][tau].append(r.mean_shortfall_ticks)
            grid["value_add_vwap"][tau].append(r.value_add_vs_vwap_ticks)
            # Same per-decision fills, three model sizing rules (agent sets side+timing).
            grid["size_agent"][tau].append(size_weighted_shortfall(r.fills, "agent"))
            grid["size_confidence"][tau].append(size_weighted_shortfall(r.fills, "confidence"))
            grid["size_inverse_vol"][tau].append(size_weighted_shortfall(r.fills, "inverse_vol"))
        row = " ".join(
            f"frt={f}:fill={grid['fill_rate'][tau][i]:.0%}/med={grid['median_shortfall'][tau][i]:+.1f}t"
            for i, f in enumerate(FILL_TARGETS)
        )
        print(f"  tau={tau:>2}: {row}")

    _plot_overview(market_dir.name, grid)
    _plot_sizing(market_dir.name, grid)
    _summarize_best_by_tau(market_dir.name, grid)
    return market_dir.name, grid


def _summarize_best_by_tau(label: str, grid: dict) -> None:
    """For each τ, the best sizing rule (best size-weighted shortfall over fill_rate_target).

    Prints a table and draws a grouped bar chart `agent_best_by_tau_<market>.png`:
    per τ, each rule's best-achievable size-weighted shortfall, winner annotated.
    """
    print(f"\n  --- {label}: best sizing rule per tau (max size-weighted shortfall over fill targets) ---")
    print(f"  {'tau':>3} | {'agent':>18} | {'confidence':>18} | {'inverse-vol':>18} | winner")
    best: dict[str, list[float]] = {r: [] for r in SIZING_RULES}
    for tau in TAUS:
        cells = []
        for rule in SIZING_RULES:
            vals = grid[rule][tau]
            i = int(np.argmax(vals))
            best[rule].append(vals[i])
            cells.append(f"{vals[i]:+.2f}t @frt{FILL_TARGETS[i]:.1f}")
        winner = max(SIZING_RULES, key=lambda r: best[r][-1])
        print(f"  {tau:>3} | {cells[0]:>18} | {cells[1]:>18} | {cells[2]:>18} | {RULE_LABEL[winner]}")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TAUS))
    width = 0.26
    colors = {"size_agent": "tab:gray", "size_confidence": "tab:blue", "size_inverse_vol": "tab:red"}
    for off, rule in zip((-width, 0, width), SIZING_RULES, strict=False):
        ax.bar(x + off, best[rule], width, label=RULE_LABEL[rule], color=colors[rule])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ={t}" for t in TAUS])
    ax.set_ylabel("best size-weighted mean shortfall (ticks)")
    ax.set_title(f"{label} — best of each sizing rule per τ (higher = better; agent side, model size)")
    ax.legend()
    fig.tight_layout()
    path = REPORTS / f"agent_best_by_tau_{label.split()[0]}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def _plot_overview(label: str, grid: dict) -> None:
    """General execution metrics vs parameters (one line per τ)."""
    panels = [
        ("fill_rate", "fill rate", "fraction"),
        ("median_shortfall", "median shortfall (honest headline)", "ticks"),
        ("mean_shortfall", "mean shortfall (chase-tail dragged)", "ticks"),
        ("value_add_vwap", "value-add vs VWAP (mean)", "ticks"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    cmap = plt.get_cmap("viridis")
    for ax, (key, title, ylab) in zip(axes, panels, strict=False):
        for i, tau in enumerate(TAUS):
            ax.plot(
                FILL_TARGETS, grid[key][tau], marker="o", markersize=4,
                color=cmap(i / max(len(TAUS) - 1, 1)), label=f"τ={tau}",
            )
        if "shortfall" in key or "value" in key:
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fill_rate_target")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
    fig.suptitle(
        f"{label} — agent supplies side, our model supplies execution (ℓ*); metrics vs parameters",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = REPORTS / f"agent_sweep_{label.split()[0]}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def _plot_sizing(label: str, grid: dict) -> None:
    """Compare the three model sizing rules: size-weighted mean shortfall, one panel per τ."""
    rules = [
        ("size_agent", "agent size (|Δpos|)", "tab:gray"),
        ("size_confidence", "confidence (fill_prob·ℓ*)", "tab:blue"),
        ("size_inverse_vol", "inverse-vol (1/σ̄)", "tab:red"),
    ]
    ncols = 3
    nrows = math.ceil(len(TAUS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, tau in zip(axes, TAUS, strict=False):
        for key, name, color in rules:
            ax.plot(FILL_TARGETS, grid[key][tau], marker="o", markersize=4, color=color, label=name)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"τ = {tau} min", fontsize=10)
        ax.set_xlabel("fill_rate_target")
        ax.set_ylabel("size-weighted mean shortfall (ticks)")
        ax.legend(fontsize=7)
    for ax in axes[len(TAUS):]:
        ax.set_visible(False)
    fig.suptitle(
        f"{label} — model order-sizing rules compared (+ve = beat arrival, size-weighted)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = REPORTS / f"agent_sizing_{label.split()[0]}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", action="append", default=[], help="market substring (repeatable)")
    args = ap.parse_args()
    for substring in args.market or ["Gold"]:
        sweep_market(substring)


if __name__ == "__main__":
    main()
