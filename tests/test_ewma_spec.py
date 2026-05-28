"""Spec-derived EWMA/EWMV oracle (Algorithm 1) vs the team's implementation.

Unlike tests/test_ewma.py — whose ``_brute_force`` mirrors the *source loop* and
so can only prove "code matches its own docstring" — this oracle is transcribed
directly from Algorithm 1 of the assignment spec
(``TermProject2_OrderExecution.pdf``, §5). The spec updates step ``j`` with the
*prior* interval ``eta_{j-1}`` ("to avoid forward looking calculation"), which in
0-based numpy is ``eta[j-2]``. The team's loop reads ``eta[j-1]`` instead, so:

  * ``eta[1]`` (the 2nd observation) is never folded in, and
  * every stored output leads the spec series by one observation.

The numeric-match test is therefore marked ``xfail(strict=True)``: it documents
the gap today and will flip to a hard failure the day the loop is corrected to
``eta[j-2]``. See notes/component-review.md ("Stream A findings").
"""

from __future__ import annotations

import numpy as np
import pytest

from regime import ewma_ewmv


def _spec_ewma_ewmv(eta, half_life: int) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1, transcribed verbatim (1-based ``j``; ``eta_{j-1}`` == ``eta[j-2]``).

    Output position ``j-1`` (0-based) holds ``ewma_j`` / ``ewmv_j``. Positions 0
    and 1 stay NaN to match the team's output convention, so the two series are
    directly comparable.
    """
    eta = np.asarray(eta, dtype=float)
    n = len(eta)
    out_ewma = np.full(n, np.nan)
    out_ewmv = np.full(n, np.nan)
    if n < 3:
        return out_ewma, out_ewmv

    lam = 2.0 ** (-1.0 / half_life)

    # j == 2: sumWX = eta_1 = eta[0]  (ewma_2 -> out[1], left NaN by convention)
    sum_w = 1.0
    sum_wx = float(eta[0])
    ewma = sum_wx / sum_w
    sum_wss = (float(eta[0]) - ewma) ** 2  # == 0

    # j = 3 … n: fold in the prior interval eta_{j-1} == eta[j-2]
    for j in range(3, n + 1):
        x = float(eta[j - 2])
        sum_w = lam * sum_w + 1.0
        sum_wx = lam * sum_wx + x
        ewma = sum_wx / sum_w
        sum_wss = lam * sum_wss + (x - ewma) ** 2
        out_ewma[j - 1] = ewma
        out_ewmv[j - 1] = (sum_wss / sum_w) ** 0.5

    return out_ewma, out_ewmv


@pytest.mark.xfail(
    reason="regime.ewma_ewmv reads eta[j-1] (eta_j); Algorithm 1 specifies "
    "eta_{j-1}=eta[j-2], so eta[1] is dropped and the series leads spec by one "
    "step. Pending team confirmation — see notes/component-review.md",
    strict=True,
)
def test_ewma_matches_spec_algorithm1() -> None:
    """Team recursion must equal the spec oracle. Currently it does not."""
    rng = np.random.default_rng(2024)
    eta = rng.standard_normal(60)
    actual_ewma, actual_ewmv = ewma_ewmv(eta, half_life=10)
    spec_ewma, spec_ewmv = _spec_ewma_ewmv(eta, half_life=10)
    np.testing.assert_allclose(actual_ewma[2:], spec_ewma[2:], rtol=1e-12)
    np.testing.assert_allclose(actual_ewmv[2:], spec_ewmv[2:], rtol=1e-12)


def test_eta1_is_ignored_by_current_impl() -> None:
    """Current behavior: eta[1] is the one observation the loop never reads.

    Perturbing only eta[1] leaves every output unchanged — a documented,
    intentional-looking consequence of the spec deviation above, not a silent
    accident. If this ever starts failing, the loop's input index has changed.
    """
    rng = np.random.default_rng(7)
    eta = rng.standard_normal(40)
    base_ewma, base_ewmv = ewma_ewmv(eta, half_life=12)

    perturbed = eta.copy()
    perturbed[1] += 1000.0  # huge change to the single ignored observation
    pert_ewma, pert_ewmv = ewma_ewmv(perturbed, half_life=12)

    np.testing.assert_array_equal(base_ewma, pert_ewma)
    np.testing.assert_array_equal(base_ewmv, pert_ewmv)


def test_spec_oracle_does_depend_on_eta1() -> None:
    """Guard the oracle itself: the spec recursion *does* fold in eta[1]."""
    rng = np.random.default_rng(7)
    eta = rng.standard_normal(40)
    base_ewma, _ = _spec_ewma_ewmv(eta, half_life=12)

    perturbed = eta.copy()
    perturbed[1] += 1000.0
    pert_ewma, _ = _spec_ewma_ewmv(perturbed, half_life=12)

    assert not np.array_equal(base_ewma[2:], pert_ewma[2:])
