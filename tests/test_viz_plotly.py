"""Smoke tests for the Plotly figure builders used by the Streamlit dashboard.

These assert the builders return Figures on synthetic inputs (and degrade
gracefully on empty inputs); they do not validate visual styling.
"""

from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import viz_plotly as viz


@pytest.fixture
def t_list():
    return list(pd.date_range("2024-01-01", periods=300, freq="5min"))


def test_volume_fig_returns_figure(t_list):
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    daily = pd.DataFrame(
        {"traded_mins": rng.integers(100, 1440, 30)}, index=days
    )
    daily["is_active"] = daily["traded_mins"] >= 0.90 * daily["traded_mins"].max()
    vol_daily = pd.Series(rng.integers(1, 1000, 30), index=days)
    fig = viz.volume_fig(daily, vol_daily, "TEST")
    assert isinstance(fig, go.Figure)


def test_volume_fig_empty():
    assert isinstance(viz.volume_fig(pd.DataFrame(), pd.Series(dtype=float), "X"), go.Figure)


def test_regime_panel_fig_with_warmup_nans(t_list):
    rng = np.random.default_rng(1)
    arr = rng.standard_normal(len(t_list))
    arr[:2] = np.nan  # EWMA warm-up positions
    fig = viz.regime_panel_fig(t_list, arr, 3, "EWMA Range")
    assert isinstance(fig, go.Figure)


def test_regime_panel_fig_all_nan(t_list):
    arr = np.full(len(t_list), np.nan)
    assert isinstance(viz.regime_panel_fig(t_list, arr, 3, "Δx"), go.Figure)


def test_backtest_hist_fig():
    rng = np.random.default_rng(2)
    strat = list(rng.standard_normal(500))
    vwap = list(rng.standard_normal(500))
    fig = viz.backtest_hist_fig(strat, vwap, "buy", "TEST")
    assert isinstance(fig, go.Figure)
    assert isinstance(viz.backtest_hist_fig([], [], "sell", "TEST"), go.Figure)


def test_epdf_heatmap_fig():
    M, N, K = 3, 3, 3
    counts_RU = {(m, n, k): Counter({m + n: 10, m + n + 1: 5})
                 for m in range(1, M + 1) for n in range(1, N + 1) for k in range(1, K + 1)}
    counts_RD = {(m, n, k): Counter({n: 8}) for (m, n, k) in counts_RU}
    fig = viz.epdf_heatmap_fig(counts_RU, counts_RD, M, N, K)
    assert isinstance(fig, go.Figure)
    assert isinstance(viz.epdf_heatmap_fig({}, {}, M, N, K), go.Figure)


def test_pareto_fig():
    points = [(0.4, 0.82, 0.10), (0.6, 0.71, 0.28), (0.8, 0.55, 0.41)]
    fig = viz.pareto_fig(points, "buy", current_target=0.6)
    assert isinstance(fig, go.Figure)
    assert isinstance(viz.pareto_fig([], "buy"), go.Figure)
