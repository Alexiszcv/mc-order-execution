"""Tests for `order_mgmt.strategy`: ℓ* pickers and chase models."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from order_mgmt.strategy import (
    chase_price,
    pick_ell_star,
    pick_ell_star_cost_aware,
    pick_ell_star_random,
    simulate_early_chase,
)

# Worked example from the chat / project docs:
# 20 windows with R_U distribution {2:3, 3:4, 4:4, 5:5, 6:2, 7:1, 8:1}.
# Survival: P(≥8)=.05, ≥7=.10, ≥6=.20, ≥5=.45, ≥4=.65, ≥3=.85, ≥2=1.00.
_EPDF = Counter({2: 3, 3: 4, 4: 4, 5: 5, 6: 2, 7: 1, 8: 1})


def _survival(epdf: Counter, ell: int) -> float:
    """P(R ≥ ell) for a count histogram — reference used to check the picker."""
    total = sum(epdf.values())
    return sum(c for k, c in epdf.items() if k >= ell) / total


def test_pick_ell_star_worked_example() -> None:
    assert pick_ell_star(_EPDF, 0.60) == 4


def test_pick_ell_star_at_exact_threshold() -> None:
    assert pick_ell_star(_EPDF, 0.65) == 4  # P(≥4) = 0.65 exactly


def test_pick_ell_star_just_above_threshold_drops_one() -> None:
    assert pick_ell_star(_EPDF, 0.66) == 3  # 0.66 > 0.65; falls to ℓ=3 (P=0.85)


def test_pick_ell_star_zero_target_returns_max_observed() -> None:
    assert pick_ell_star(Counter({2: 3, 5: 5, 8: 1}), 0.0) == 8


def test_pick_ell_star_target_above_one_returns_zero() -> None:
    assert pick_ell_star(_EPDF, 1.5) == 0


def test_pick_ell_star_empty_epdf_returns_zero() -> None:
    assert pick_ell_star(Counter(), 0.5) == 0


def test_pick_ell_star_single_bucket() -> None:
    epdf = Counter({5: 100})
    assert pick_ell_star(epdf, 0.5) == 5
    assert pick_ell_star(epdf, 1.0) == 5


def test_pick_ell_star_monotone_in_target() -> None:
    """As target increases, ℓ* should not increase."""
    prev = pick_ell_star(_EPDF, 0.0)
    for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        cur = pick_ell_star(_EPDF, t)
        assert cur <= prev
        prev = cur


@pytest.mark.parametrize("target", [0.10, 0.20, 0.45, 0.60, 0.65, 0.80, 0.95])
def test_pick_ell_star_achieves_target_and_is_largest(target: float) -> None:
    """ℓ* meets the target, and ℓ*+1 would fall below it (it is the largest such ℓ)."""
    ell = pick_ell_star(_EPDF, target)
    assert _survival(_EPDF, ell) >= target
    assert _survival(_EPDF, ell + 1) < target


# --- cost-aware picker (hand-computed against _EPDF survival) -----------------
# objective(ℓ) = P(R≥ℓ)·ℓ - (1-P(R≥ℓ))·chase_cost.


def test_cost_aware_no_chase_cost_maximises_expected_savings() -> None:
    # chase_cost=0 → maximise p·ℓ: ℓ=4 gives .65·4=2.6, the max over the support.
    assert pick_ell_star_cost_aware(_EPDF, 0.0) == 4


def test_cost_aware_moderate_cost_pulls_in() -> None:
    # chase_cost=2 → ℓ=3 scores .85·3 - .15·2 = 2.25, the argmax.
    assert pick_ell_star_cost_aware(_EPDF, 2.0) == 3


def test_cost_aware_heavy_cost_is_conservative() -> None:
    # chase_cost=10 → ℓ=2 scores 1.0·2 - 0 = 2.0, beating every higher ℓ.
    assert pick_ell_star_cost_aware(_EPDF, 10.0) == 2


def test_cost_aware_empty_returns_zero() -> None:
    assert pick_ell_star_cost_aware(Counter(), 1.0) == 0


# --- random picker (ablation control) -----------------------------------------


def test_random_picker_within_support() -> None:
    """Every draw stays in [0, max ℓ in the support] — the same band as the real picker."""
    rng = np.random.default_rng(0)
    max_ell = max(_EPDF.keys())
    for _ in range(1000):
        ell = pick_ell_star_random(_EPDF, rng)
        assert 0 <= ell <= max_ell


def test_random_picker_seeded_is_deterministic() -> None:
    """Same seed → same sequence of ℓ* (reproducibility of the ablation control)."""
    a = [pick_ell_star_random(_EPDF, np.random.default_rng(7)) for _ in range(5)]
    b = [pick_ell_star_random(_EPDF, np.random.default_rng(7)) for _ in range(5)]
    assert a == b


def test_random_picker_spans_the_band() -> None:
    """Over many draws it should actually use the range, not collapse to one value."""
    rng = np.random.default_rng(1)
    seen = {pick_ell_star_random(_EPDF, rng) for _ in range(2000)}
    assert 0 in seen and max(_EPDF.keys()) in seen and len(seen) > 3


def test_random_picker_empty_returns_zero() -> None:
    assert pick_ell_star_random(Counter(), np.random.default_rng(0)) == 0


def test_random_picker_degenerate_zero_support() -> None:
    # max_ell <= 0 → only ℓ=0 is possible.
    assert pick_ell_star_random(Counter({0: 5}), np.random.default_rng(0)) == 0


# --- chase_price --------------------------------------------------------------


def test_chase_price_close_returns_close() -> None:
    assert chase_price(100.0, 0.1, policy="close", window_close=100.5) == 100.5


def test_chase_price_mid_returns_window_mid() -> None:
    assert chase_price(100.0, 0.1, policy="mid", window_high=101.0, window_low=99.0) == 100.0


def test_chase_price_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        chase_price(100.0, 0.1, policy="vwap", window_close=100.5)  # type: ignore[arg-type]


# --- simulate_early_chase -----------------------------------------------------
# Sell: open=100, ell_star=5, tick=0.1 → limit=100.5; trigger=3 → bail=99.7.


def test_early_chase_sell_fills_at_limit() -> None:
    price, filled = simulate_early_chase(
        "sell",
        100.0,
        5,
        0.1,
        highs=[100.2, 100.6],
        lows=[99.9, 100.0],
        closes=[100.1, 100.5],
        trigger_ticks=3,
    )
    assert filled is True
    assert price == pytest.approx(100.5)


def test_early_chase_sell_bails_at_trigger() -> None:
    price, filled = simulate_early_chase(
        "sell",
        100.0,
        5,
        0.1,
        highs=[100.1, 100.2],
        lows=[99.95, 99.6],
        closes=[100.0, 99.7],
        trigger_ticks=3,
    )
    assert filled is False
    assert price == pytest.approx(99.7)


def test_early_chase_sell_falls_through_to_close() -> None:
    price, filled = simulate_early_chase(
        "sell",
        100.0,
        5,
        0.1,
        highs=[100.1, 100.2],
        lows=[99.8, 99.85],
        closes=[100.0, 100.05],
        trigger_ticks=3,
    )
    assert filled is False
    assert price == pytest.approx(100.05)


def test_early_chase_same_bar_tie_favors_limit() -> None:
    # One bar reaches both the limit (100.5) and the bail (99.7): limit wins the tie.
    price, filled = simulate_early_chase(
        "sell",
        100.0,
        5,
        0.1,
        highs=[100.6],
        lows=[99.6],
        closes=[100.0],
        trigger_ticks=3,
    )
    assert filled is True
    assert price == pytest.approx(100.5)


def test_early_chase_buy_fills_at_limit() -> None:
    # Buy: open=100, ell_star=5 → limit=99.5; trigger=3 → bail=100.3.
    price, filled = simulate_early_chase(
        "buy",
        100.0,
        5,
        0.1,
        highs=[100.1, 100.0],
        lows=[99.8, 99.4],
        closes=[99.9, 99.5],
        trigger_ticks=3,
    )
    assert filled is True
    assert price == pytest.approx(99.5)


def test_early_chase_buy_bails_at_trigger() -> None:
    price, filled = simulate_early_chase(
        "buy",
        100.0,
        5,
        0.1,
        highs=[100.1, 100.4],
        lows=[99.8, 99.7],
        closes=[100.0, 100.3],
        trigger_ticks=3,
    )
    assert filled is False
    assert price == pytest.approx(100.3)


def test_early_chase_rejects_nonpositive_trigger() -> None:
    with pytest.raises(ValueError):
        simulate_early_chase(
            "sell",
            100.0,
            5,
            0.1,
            highs=[100.0],
            lows=[100.0],
            closes=[100.0],
            trigger_ticks=0,
        )
