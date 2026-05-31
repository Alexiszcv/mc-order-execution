"""Tests for the ablation ladder in `scripts/ablation.py`.

Two things matter: (1) the conditional rung must be the SAME decision path as the trusted
v2 backtest (else the comparison is meaningless), and (2) all rungs must be scored on an
identical decision set (else their means aren't comparable). A day-truncation invariant
also guards no-lookahead across every rung — including persistence and the seeded random
control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from epdf import _load_1min
from order_mgmt.backtest import run_backtest_rolling
from plot_volume import _compute_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ablation  # noqa: E402

GOLD_CSV = ROOT / "data" / "Gold" / "GCM24.csv"
pytestmark = pytest.mark.skipif(not GOLD_CSV.exists(), reason="Gold data not present")

TGT = 0.6


def _load() -> tuple[pd.DataFrame, float, list]:
    df = _load_1min(str(GOLD_CSV))
    _, tick, proper_days, _, _ = _compute_stats(df)
    return df, tick, proper_days


@pytest.mark.parametrize("side", ["sell", "buy"])
def test_conditional_rung_matches_v2(side: str) -> None:
    """conditional == run_backtest_rolling on n_decisions / n_filled / mean slippage."""
    df, tick, days = _load()
    res = ablation.run_ablation(df, 5, tick, days, side, targets=[TGT])
    c = res["conditional"][TGT]
    v2 = run_backtest_rolling(
        df,
        tau=5,
        tick=tick,
        proper_days=days,
        side=side,
        fill_rate_target=TGT,
        half_life=20,
        M=3,
        N=3,
        K=3,
        j_start=200,
    )
    assert c.n_decisions == v2.n_decisions
    assert c.n_filled == v2.n_filled
    assert abs(c.avg - v2.avg_slippage_ticks) < 1e-9


def test_all_rungs_share_decision_set() -> None:
    """Every rung is scored on exactly the same windows (same n), so means compare."""
    df, tick, days = _load()
    res = ablation.run_ablation(df, 5, tick, days, "sell", targets=[TGT])
    n = res["conditional"][TGT].n_decisions
    assert n > 0
    assert res["unconditional"][TGT].n_decisions == n
    for rung in ("persistence", "random", "vwap"):
        assert res[rung].n_decisions == n


def test_ablation_no_lookahead_via_truncation() -> None:
    """Truncating the data at a day boundary must not change earlier decisions — for ANY
    rung (conditional, unconditional, persistence, and the seeded random control)."""
    df, tick, days = _load()
    if len(days) < 4:
        pytest.skip("not enough active days to truncate meaningfully")

    cutoff_end = pd.Timestamp(days[int(len(days) * 0.75)]).normalize() + pd.Timedelta(days=1)
    df_short = df[df.index < cutoff_end]
    days_short = [d for d in days if pd.Timestamp(d).normalize() < cutoff_end]

    full = ablation.run_ablation(df, 5, tick, days, "sell", targets=[TGT])
    short = ablation.run_ablation(df_short, 5, tick, days_short, "sell", targets=[TGT])

    def slips(res: dict, rung: str) -> list[float]:
        r = res[rung][TGT] if rung in ("conditional", "unconditional") else res[rung]
        return r.slips

    for rung in ("conditional", "unconditional", "persistence", "random", "vwap"):
        f, s = slips(full, rung), slips(short, rung)
        k = min(len(f), len(s))
        assert k > 0
        assert f[:k] == s[:k], f"{rung} decisions changed under truncation (look-ahead)"
