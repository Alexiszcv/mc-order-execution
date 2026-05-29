"""Tests for `order_mgmt.strategy.pick_ell_star`."""

from __future__ import annotations

from collections import Counter

from order_mgmt.strategy import pick_ell_star

# Worked example from the chat / project docs:
# 20 windows with R_U distribution {2:3, 3:4, 4:4, 5:5, 6:2, 7:1, 8:1}.
# Survival: P(≥8)=.05, ≥7=.10, ≥6=.20, ≥5=.45, ≥4=.65, ≥3=.85, ≥2=1.00.
_EPDF = Counter({2: 3, 3: 4, 4: 4, 5: 5, 6: 2, 7: 1, 8: 1})


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
