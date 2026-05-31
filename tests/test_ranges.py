"""Range identity tests: R = R_U + R_D (up to ±1 tick rounding).

Exercises the team's `compute_all_ranges` in `src/epdf.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ranges import compute_all_ranges


def _synth_window(date: str, opens, highs, lows, closes, vols) -> pd.DataFrame:
    times = pd.date_range(f"{date} 09:30:00", periods=len(opens), freq="1min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=times,
    )


def test_range_identity_handcrafted() -> None:
    """Known window with hand-computed R, R_U, R_D."""
    df = _synth_window(
        "2024-01-02",
        opens=[100.0, 100.3, 100.5, 100.4, 100.2],
        highs=[100.5, 100.7, 100.9, 100.6, 100.5],
        lows=[99.8, 100.1, 100.3, 100.2, 100.0],
        closes=[100.3, 100.5, 100.4, 100.2, 100.1],
        vols=[10] * 5,
    )
    # H = 100.9, L = 99.8, O = 100.0, tick = 0.1
    # R = 1.1 → 11 ticks, R_U = 0.9 → 9 ticks, R_D = 0.2 → 2 ticks
    _, ell_r, ell_u, ell_d, *_ = compute_all_ranges(
        df, tau=5, tick=0.1, proper_days_list=[pd.Timestamp("2024-01-02")]
    )
    assert len(ell_r) == 1
    assert ell_r[0] == 11
    assert ell_u[0] == 9
    assert ell_d[0] == 2
    assert ell_r[0] == ell_u[0] + ell_d[0]


def test_range_identity_random_within_one_tick() -> None:
    """For random OHLC bars, R = R_U + R_D up to ±1 tick rounding."""
    rng = np.random.default_rng(42)
    bars_per_day, days = 60, 5
    dfs = []
    for d in range(days):
        date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        times = pd.date_range(date.replace(hour=9, minute=30), periods=bars_per_day, freq="1min")
        opens = 100.0 + rng.standard_normal(bars_per_day).cumsum() * 0.1
        closes = opens + rng.standard_normal(bars_per_day) * 0.05
        highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(bars_per_day)) * 0.1
        lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(bars_per_day)) * 0.1
        dfs.append(
            pd.DataFrame(
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": rng.integers(10, 100, bars_per_day),
                },
                index=times,
            )
        )
    df = pd.concat(dfs)
    days_list = [pd.Timestamp("2024-01-02") + pd.Timedelta(days=d) for d in range(days)]

    _, ell_r, ell_u, ell_d, *_ = compute_all_ranges(df, tau=5, tick=0.01, proper_days_list=days_list)
    assert len(ell_r) > 0
    diffs = [abs(r - (u + d)) for r, u, d in zip(ell_r, ell_u, ell_d, strict=True)]
    assert max(diffs) <= 1, f"identity violated by {max(diffs)} ticks"


def test_range_nonneg() -> None:
    """All three quantities must be non-negative."""
    df = _synth_window(
        "2024-01-02",
        opens=[100.0, 100.0, 100.0, 100.0, 100.0],
        highs=[100.5, 100.7, 100.9, 100.6, 100.5],
        lows=[99.8, 100.0, 100.0, 100.0, 100.0],
        closes=[100.3, 100.5, 100.4, 100.2, 100.1],
        vols=[10] * 5,
    )
    _, ell_r, ell_u, ell_d, *_ = compute_all_ranges(
        df, tau=5, tick=0.1, proper_days_list=[pd.Timestamp("2024-01-02")]
    )
    assert ell_r[0] >= 0
    assert ell_u[0] >= 0
    assert ell_d[0] >= 0
