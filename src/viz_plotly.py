"""Interactive Plotly figure builders for the Streamlit dashboard.

Pure functions: each takes already-computed series/arrays and returns a
``plotly.graph_objects.Figure``. No Streamlit import here, so the builders are
unit-testable in isolation. The quantile-band logic mirrors
``regime._colored_panel`` (percentiles at 100*k/n_states) so the interactive
regime panels match the matplotlib reference exactly.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

# Dark-theme template used across every figure so they sit cleanly on the
# Streamlit dark background.
_TEMPLATE = "plotly_dark"
_BAND_ALPHA = 0.16


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#888"))
    fig.update_layout(template=_TEMPLATE, xaxis_visible=False, yaxis_visible=False)
    return fig


def _state_bounds(values: np.ndarray, n_states: int):
    """Return (bounds, colors, (y_lo, y_hi)) for quantile bands.

    `bounds` has length n_states+1, clipped to the padded data range (matching
    regime._colored_panel). `colors` is a RdYlGn sample, one rgb per state.
    """
    valid = values[~np.isnan(values)]
    qs = np.percentile(valid, [100.0 * k / n_states for k in range(1, n_states)])
    bounds = np.concatenate([[-np.inf], qs, [np.inf]])
    if n_states == 1:
        colors = sample_colorscale("RdYlGn", [0.5])
    else:
        colors = sample_colorscale("RdYlGn", [k / (n_states - 1) for k in range(n_states)])
    ymin, ymax = float(valid.min()), float(valid.max())
    pad = max((ymax - ymin) * 0.05, 1e-9)
    return bounds, colors, (ymin - pad, ymax + pad)


def volume_fig(daily, vol_daily, ticker: str) -> go.Figure:
    """Daily traded volume; active days (>=90% of max traded minutes) shaded green.

    `daily` is the DataFrame from plot_volume._compute_stats (index=day, column
    `is_active`); `vol_daily` is the per-day summed volume Series (>0 only).
    """
    if vol_daily is None or len(vol_daily) == 0:
        return _empty("No volume data")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(vol_daily.index),
            y=vol_daily.to_numpy(),
            marker_color="#4f8fd6",
            name="daily volume",
            hovertemplate="%{x|%Y-%m-%d}<br>volume=%{y:,.0f}<extra></extra>",
        )
    )
    # One green span per contiguous run of active days (keeps the shape count low).
    active = daily["is_active"]
    one_day = (daily.index[1] - daily.index[0]) if len(daily.index) > 1 else None
    start = None
    prev = None
    for day, is_act in active.items():
        if is_act and start is None:
            start = day
        if (not is_act) and start is not None:
            fig.add_vrect(x0=start, x1=prev, fillcolor="#4CAF50", opacity=0.12,
                          line_width=0, layer="below")
            start = None
        prev = day
    if start is not None and one_day is not None:
        fig.add_vrect(x0=start, x1=prev + one_day, fillcolor="#4CAF50", opacity=0.12,
                      line_width=0, layer="below")

    fig.update_layout(
        template=_TEMPLATE, title=f"{ticker} — daily traded volume",
        xaxis_title="date", yaxis_title="volume", bargap=0.1,
        margin=dict(l=50, r=20, t=50, b=40), height=380, showlegend=False,
    )
    return fig


def regime_panel_fig(t_list, arr, n_states: int, label: str) -> go.Figure:
    """Time-series line + quantile-band coloring, with a marginal histogram.

    `arr` is the value series (EWMA range / EWMA volume / Δx); leading NaNs (EWMA
    warm-up) are tolerated. Bands are coloured RdYlGn by quantile state.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return _empty(f"{label}: no data")

    bounds, colors, (y_lo, y_hi) = _state_bounds(arr, n_states)
    mask = ~np.isnan(arr)
    t_valid = [t_list[i] for i in range(len(t_list)) if mask[i]]
    v_valid = arr[mask]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.82, 0.18],
        horizontal_spacing=0.01,
    )
    # Coloured quantile bands behind the line, on both the series and histogram.
    for k in range(n_states):
        lo = max(float(bounds[k]), y_lo)
        hi = min(float(bounds[k + 1]), y_hi)
        for col in (1, 2):
            fig.add_hrect(y0=lo, y1=hi, fillcolor=colors[k], opacity=_BAND_ALPHA,
                          line_width=0, layer="below", row=1, col=col)

    fig.add_trace(
        go.Scatter(x=t_valid, y=v_valid, mode="lines",
                   line=dict(color="#EDEDED", width=0.8), name=label,
                   hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(y=v_valid, nbinsy=60, marker_color="#8a93a6",
                     orientation="h", showlegend=False,
                     hovertemplate="%{y:.2f}: %{x}<extra></extra>"),
        row=1, col=2,
    )
    fig.update_yaxes(range=[y_lo, y_hi], row=1, col=1)
    fig.update_layout(
        template=_TEMPLATE, title=label, showlegend=False,
        margin=dict(l=50, r=10, t=50, b=40), height=330, bargap=0.02,
    )
    fig.update_xaxes(title_text="time", row=1, col=1)
    fig.update_xaxes(title_text="count", row=1, col=2)
    return fig


def backtest_hist_fig(strat_slip, vwap_slip, side: str, ticker: str) -> go.Figure:
    """Overlaid slippage histograms: strategy (v2, no-lookahead) vs VWAP."""
    if not strat_slip and not vwap_slip:
        return _empty(f"{side}: no backtest data")
    fig = go.Figure()
    if strat_slip:
        fig.add_trace(go.Histogram(x=strat_slip, nbinsx=50, opacity=0.6,
                                   name="Strategy (v2)", marker_color="#4CAF50"))
    if vwap_slip:
        fig.add_trace(go.Histogram(x=vwap_slip, nbinsx=50, opacity=0.6,
                                   name="VWAP", marker_color="#ED9E3B"))
    fig.add_vline(x=0, line_dash="dash", line_color="#cccccc", line_width=1)
    fig.update_layout(
        template=_TEMPLATE, barmode="overlay",
        title=f"{ticker} — {side} (slippage vs open, ticks)",
        xaxis_title="ticks (positive = beats open)", yaxis_title="count",
        margin=dict(l=50, r=20, t=50, b=40), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    return fig


def _cell_grids(counts: dict[tuple[int, int, int], Counter], M: int, N: int, k: int):
    """For a fixed direction state k, return (mean_grid, count_grid) over (m rows, n cols)."""
    mean_grid = np.full((M, N), np.nan)
    count_grid = np.zeros((M, N), dtype=int)
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            ctr = counts.get((m, n, k))
            if not ctr:
                continue
            total = sum(ctr.values())
            mean_grid[m - 1, n - 1] = sum(ell * c for ell, c in ctr.items()) / total
            count_grid[m - 1, n - 1] = total
    return mean_grid, count_grid


def epdf_heatmap_fig(counts_RU, counts_RD, M: int, N: int, K: int) -> go.Figure:
    """Regime grid as heatmaps: rows = R_U / R_D, cols = direction state k.

    Each heatmap is m (volume state, y) by n (range state, x), coloured by the
    mean range in ticks; hover shows the observation count per cell.
    """
    if not counts_RU and not counts_RD:
        return _empty("No populated regime cells (lower j_start or use fewer states)")

    fig = make_subplots(
        rows=2, cols=K, shared_xaxes=True, shared_yaxes=True,
        column_titles=[f"k={k}" for k in range(1, K + 1)],
        row_titles=["mean R<sub>U</sub>", "mean R<sub>D</sub>"],
        horizontal_spacing=0.04, vertical_spacing=0.10,
    )
    x_labels = [f"n={n}" for n in range(1, N + 1)]
    y_labels = [f"m={m}" for m in range(1, M + 1)]
    for row, counts in ((1, counts_RU), (2, counts_RD)):
        for k in range(1, K + 1):
            mean_grid, count_grid = _cell_grids(counts, M, N, k)
            fig.add_trace(
                go.Heatmap(
                    z=mean_grid, x=x_labels, y=y_labels,
                    customdata=count_grid, coloraxis="coloraxis",
                    hovertemplate="m=%{y}, n=%{x}<br>mean=%{z:.2f}t<br>n_obs=%{customdata}<extra></extra>",
                ),
                row=row, col=k,
            )
    fig.update_layout(
        template=_TEMPLATE, height=520,
        coloraxis=dict(colorscale="Viridis", colorbar=dict(title="ticks")),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def pareto_fig(points, side: str, current_target: float | None = None) -> go.Figure:
    """Fill-target sweep: realized fill-rate (x) vs avg slippage in ticks (y).

    `points` is a list of (fill_target, fill_rate, avg_slippage) tuples.
    """
    if not points:
        return _empty("Enable and run the sweep to see the Pareto curve")
    pts = sorted(points, key=lambda p: p[0])
    targets = [p[0] for p in pts]
    fills = [p[1] for p in pts]
    slips = [p[2] for p in pts]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fills, y=slips, mode="lines+markers+text",
            text=[f"{t:.2f}" for t in targets], textposition="top center",
            marker=dict(size=9, color=targets, colorscale="Viridis",
                        colorbar=dict(title="fill<br>target"), showscale=True),
            line=dict(color="#888", width=1),
            hovertemplate="target=%{text}<br>fill-rate=%{x:.1%}<br>avg slip=%{y:+.2f}t<extra></extra>",
            name="sweep",
        )
    )
    if current_target is not None:
        match = next((p for p in pts if abs(p[0] - current_target) < 1e-9), None)
        if match is not None:
            fig.add_trace(
                go.Scatter(x=[match[1]], y=[match[2]], mode="markers",
                           marker=dict(size=16, color="rgba(0,0,0,0)",
                                       line=dict(color="#4CAF50", width=3)),
                           name="current", hoverinfo="skip"),
            )
    fig.update_layout(
        template=_TEMPLATE, title=f"{side} — fill-rate vs slippage tradeoff",
        xaxis_title="realized fill-rate", yaxis_title="avg slippage (ticks)",
        xaxis_tickformat=".0%", margin=dict(l=50, r=20, t=50, b=40), height=420,
        showlegend=False,
    )
    return fig
