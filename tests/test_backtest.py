"""Tests for the backtest loops in `order_mgmt.backtest`.

The rolling (v2) backtest's no-lookahead invariant is exercised by truncating the
input data and verifying that the early-window decisions are byte-identical to the
corresponding decisions in the full-data run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from epdf import _load_1min
from plot_volume import _compute_stats

from order_mgmt.backtest import run_backtest, run_backtest_rolling

ROOT = Path(__file__).resolve().parents[1]
GOLD_CSV = ROOT / "data" / "Gold" / "GCM24.csv"

pytestmark = pytest.mark.skipif(not GOLD_CSV.exists(), reason="Gold data not present")


def _load() -> tuple[pd.DataFrame, float, list]:
    df = _load_1min(str(GOLD_CSV))
    _, tick, proper_days, _, _ = _compute_stats(df)
    return df, tick, proper_days


def test_v1_runs_and_returns_nonzero_decisions() -> None:
    df, tick, days = _load()
    r = run_backtest(
        df, tau=5, tick=tick, proper_days=days, side="sell",
        fill_rate_target=0.6, half_life=20, M=3, N=3, K=3, j_start=200,
    )
    assert r.n_decisions > 0
    assert 0.0 <= r.fill_rate <= 1.0


def test_v2_runs_and_returns_nonzero_decisions() -> None:
    df, tick, days = _load()
    r = run_backtest_rolling(
        df, tau=5, tick=tick, proper_days=days, side="sell",
        fill_rate_target=0.6, half_life=20, M=3, N=3, K=3, j_start=200,
    )
    assert r.n_decisions > 0
    assert 0.0 <= r.fill_rate <= 1.0


def test_v2_no_lookahead_via_day_truncation() -> None:
    """Truncating the input at any day-boundary must not change earlier v2 decisions."""
    df, tick, days = _load()
    if len(days) < 4:
        pytest.skip("not enough active days to truncate meaningfully")

    # Keep ~75% of active days; truncate the DataFrame at the last bar of that day.
    cutoff_day = pd.Timestamp(days[int(len(days) * 0.75)]).normalize()
    cutoff_end = cutoff_day + pd.Timedelta(days=1)
    df_short = df[df.index < cutoff_end]
    days_short = [d for d in days if pd.Timestamp(d).normalize() < cutoff_end]

    full = run_backtest_rolling(
        df, tau=5, tick=tick, proper_days=days, side="sell",
        fill_rate_target=0.6, half_life=20, M=3, N=3, K=3, j_start=200,
    )
    short = run_backtest_rolling(
        df_short, tau=5, tick=tick, proper_days=days_short, side="sell",
        fill_rate_target=0.6, half_life=20, M=3, N=3, K=3, j_start=200,
    )

    k = min(len(full.slippage_ticks), len(short.slippage_ticks))
    assert k > 0
    # The first k decisions must be byte-identical: same regimes, same ePDFs, same ℓ*.
    assert full.slippage_ticks[:k] == short.slippage_ticks[:k]
    assert full.realized_prices[:k] == short.realized_prices[:k]
