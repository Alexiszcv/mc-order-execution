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
from ranges import compute_all_ranges

from order_mgmt.backtest import run_backtest, run_backtest_rolling
from order_mgmt.baselines import Side, vwap_baseline

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


def _vwap_reference_mask(
    df_1min: pd.DataFrame, t_list: list, tau: int, tick: float, side: Side
) -> tuple[list[float], list[float]]:
    """Pre-optimization O(n²) boolean-mask VWAP, kept only as the equivalence oracle.

    Mirrors the exact arithmetic `vwap_baseline` replaced so any divergence between
    the searchsorted rewrite and this reference is a real behaviour change, not noise.
    """
    dt_tau = pd.Timedelta(minutes=tau)
    opens_series = df_1min["open"]
    realized: list[float] = []
    slippage: list[float] = []
    for t in t_list:
        window = df_1min.loc[(df_1min.index >= t) & (df_1min.index < t + dt_tau)]
        if window.empty:
            continue
        typ = (window["high"] + window["low"] + window["close"]) / 3.0
        v = window["volume"]
        vwap = float(typ.mean()) if v.sum() == 0 else float((typ * v).sum() / v.sum())
        open_j = float(opens_series.loc[t])
        realized.append(vwap)
        slippage.append((vwap - open_j) / tick if side == "sell" else (open_j - vwap) / tick)
    return realized, slippage


def test_vwap_searchsorted_matches_mask_reference() -> None:
    """The searchsorted VWAP rewrite must be byte-for-byte equivalent to the mask version."""
    df, tick, days = _load()
    t_list, *_ = compute_all_ranges(df, 5, tick, days)
    # A few hundred contiguous windows is enough to prove equivalence (incl. the
    # zero-volume branch) without paying the O(n²) reference cost over the full set.
    t_sub = t_list[200:600]
    assert t_sub, "expected enough windows to test VWAP equivalence"

    for side in ("buy", "sell"):
        fast = vwap_baseline(df, t_sub, tau=5, tick=tick, side=side)
        ref_real, ref_slip = _vwap_reference_mask(df, t_sub, 5, tick, side)
        assert fast.realized_prices == pytest.approx(ref_real, rel=1e-9, abs=1e-9)
        assert fast.slippage_ticks == pytest.approx(ref_slip, rel=1e-9, abs=1e-9)


def test_v1_v2_mean_slippage_agreement() -> None:
    """Guard the headline 'v1≈v2' result on real Gold data.

    The permissive full-history v1 and the strict no-lookahead v2 must agree on mean
    slippage within 0.1 ticks at τ=5, M=N=K=3, j_start=200. If they ever diverge past
    this, lookahead bias at this config is no longer negligible and the published
    result needs revisiting — see memory/v1_v2_lookahead_finding.md.
    """
    df, tick, days = _load()
    for side in ("buy", "sell"):
        v1 = run_backtest(
            df, tau=5, tick=tick, proper_days=days, side=side,
            fill_rate_target=0.6, half_life=20, M=3, N=3, K=3, j_start=200,
        )
        v2 = run_backtest_rolling(
            df, tau=5, tick=tick, proper_days=days, side=side,
            fill_rate_target=0.6, half_life=20, M=3, N=3, K=3, j_start=200,
        )
        assert v1.n_decisions > 0 and v2.n_decisions > 0
        assert abs(v1.avg_slippage_ticks - v2.avg_slippage_ticks) <= 0.1
