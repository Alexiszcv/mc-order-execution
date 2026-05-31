"""Small-multiples overview of every market's AI agent.

One panel per `AIAgent_*.csv`: the decision price (thin grey line) and the agent's
running signed position (filled step, blue=long / red=short) on a twin axis. Shows
at a glance how each agent trades — when it builds inventory, how often it flips,
and the position range it works within.

Run: python scripts/plot_agents.py
Outputs reports/figures/agent_overview.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from order_mgmt.agent.loader import load_agent_series, trade_decisions  # noqa: E402

REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)


def _discover_agents() -> list[tuple[str, Path]]:
    """(market_name, csv_path) for every AIAgent_*.csv under data/, sorted by market."""
    found: list[tuple[str, Path]] = []
    for csv in sorted((ROOT / "data").glob("*/AIAgent_*.csv")):
        found.append((csv.parent.name, csv))
    return found


def main() -> None:
    agents = _discover_agents()
    if not agents:
        print("no AIAgent_*.csv found under data/")
        return

    ncols = 4
    nrows = math.ceil(len(agents) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (market, csv) in zip(axes, agents, strict=False):
        series = load_agent_series(csv, market=market)
        df = series.df
        n_trades = len(trade_decisions(series))

        # price (thin grey, primary axis)
        ax.plot(df.index, df["price"], color="0.55", linewidth=0.6)
        ax.set_title(f"{market}\n{n_trades} trades, pos {int(df['position'].min())}..{int(df['position'].max())}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xticks([df.index.min(), df.index.max()])

        # running position (filled step, twin axis): blue long / red short
        ax2 = ax.twinx()
        pos = df["position"].to_numpy()
        ax2.fill_between(df.index, 0, np.clip(pos, 0, None), step="post", color="steelblue", alpha=0.5, linewidth=0)
        ax2.fill_between(df.index, 0, np.clip(pos, None, 0), step="post", color="indianred", alpha=0.5, linewidth=0)
        ax2.axhline(0, color="black", linewidth=0.4)
        ax2.tick_params(labelsize=6)

    for ax in axes[len(agents):]:
        ax.set_visible(False)

    fig.suptitle("AI agents per market — price (grey) vs running position (blue long / red short)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = REPORTS / "agent_overview.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved: {path.relative_to(ROOT)} ({len(agents)} agents)")


if __name__ == "__main__":
    main()
