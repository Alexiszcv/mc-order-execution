"""Plotting helpers separated from computation modules.

Currently just the range-histogram figure used by the web viewer.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_histogram_figure(ell_r, ell_u, ell_d, tau: int, tick: float, ticker: str):
    """Three side-by-side integer histograms of R, R_U, R_D in ticks."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        (ell_r, "R", "steelblue"),
        (ell_u, "R_U", "seagreen"),
        (ell_d, "R_D", "tomato"),
    ]
    for ax, (data, label, color) in zip(axes, specs, strict=False):
        if data:
            p99 = int(np.percentile(data, 99))
            bins = range(0, p99 + 2)
            ax.set_xlim(0, p99 + 1)
        else:
            bins = 10
        ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("ticks (ℓ)")
        ax.set_ylabel("count")
        ax.margins(x=0.02)

    plt.tight_layout()
    return fig


# Backwards-compatible alias (the team's app.py imports `_build_histogram_figure`)
_build_histogram_figure = build_histogram_figure
