"""Order-placement strategy: pick the limit-order distance from an empirical ePDF.

Two kinds of decision live here, kept as separate pure functions:

* **Limit pickers** — choose ℓ* (ticks from the window open) for a passive limit order.
  - `pick_ell_star`        : largest ℓ meeting a fill-rate target (the shared contract).
  - `pick_ell_star_cost_aware` : ℓ maximising an expected-cost objective instead of a
    fixed fill-rate target.
* **Chase models** — decide the execution price when a passive limit goes UNFILLED.
  - `chase_price`          : market-execute at the window close (baseline) or its mid.
  - `simulate_early_chase` : bail out intrabar once price runs a threshold against us.

All functions are side-aware where it matters but contain no backtest/regime logic —
the backtest owns the loop and feeds these the per-window inputs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np

Side = Literal["buy", "sell"]
ChasePolicy = Literal["close", "mid"]


def pick_ell_star(epdf: Counter, fill_rate_target: float) -> int:
    """Largest tick distance ℓ* such that P(R ≥ ℓ*) ≥ `fill_rate_target`.

    Walks the histogram from the largest observed ℓ downward, cumulating probability;
    the first ℓ where the cumulative survival meets the target is the answer (since
    survival is non-decreasing as ℓ decreases).

    Returns 0 ("always-fill default") if the target can't be met or the histogram is empty.

    Parameters
    ----------
    epdf : Counter mapping ℓ (int, ticks) → count of observations.
    fill_rate_target : float in [0, 1].

    Returns
    -------
    int : ℓ* in ticks (≥ 0).
    """
    if not epdf:
        return 0
    if fill_rate_target <= 0.0:
        return int(max(epdf.keys()))
    if fill_rate_target > 1.0:
        return 0
    total = sum(epdf.values())
    if total == 0:
        return 0
    cum = 0
    for ell in sorted(epdf.keys(), reverse=True):
        cum += epdf[ell]
        if cum / total >= fill_rate_target:
            return int(ell)
    return 0


def pick_ell_star_cost_aware(epdf: Counter, chase_cost_ticks: float) -> int:
    """Pick ℓ* maximising the expected per-window edge instead of hitting a fixed
    fill-rate target.

    The model: a passive limit ℓ ticks from the open either fills — saving ℓ ticks vs a
    market order at the open — or it doesn't, and we chase, paying `chase_cost_ticks`.
    With survival ``p(ℓ) = P(R ≥ ℓ)`` the expected edge is::

        objective(ℓ) = p(ℓ)·ℓ - (1 - p(ℓ))·chase_cost_ticks

    `objective(0) = 0` (a 0-tick limit always fills and saves nothing), so the picker
    never returns something worse than market-at-open. Ties resolve to the **largest** ℓ,
    matching `pick_ell_star`'s convention.

    Parameters
    ----------
    epdf : Counter mapping ℓ (int, ticks) → count of observations.
    chase_cost_ticks : expected adverse ticks paid when the limit is not filled (≥ 0).

    Returns
    -------
    int : ℓ* in ticks (≥ 0). Returns 0 for an empty histogram.
    """
    if not epdf:
        return 0
    total = sum(epdf.values())
    if total == 0:
        return 0

    # Survival at each candidate ℓ via one descending pass (survival is non-decreasing
    # as ℓ decreases), then score the expected-edge objective and keep the best.
    best_ell = 0
    best_obj = 0.0  # objective(0) ≡ 0: a 0-tick limit always fills, saves nothing.
    cum = 0
    for ell in sorted(epdf.keys(), reverse=True):
        cum += epdf[ell]
        p_fill = cum / total
        obj = p_fill * ell - (1.0 - p_fill) * chase_cost_ticks
        if obj > best_obj or (obj == best_obj and ell > best_ell):
            best_obj = obj
            best_ell = int(ell)
    return best_ell


def pick_ell_star_random(epdf: Counter, rng: np.random.Generator) -> int:
    """Control picker: ℓ* drawn uniformly from ``[0, max ℓ in the support]``.

    A deliberately uninformed baseline for the ablation ladder — it ignores *where* the
    survival mass sits and just picks a tick distance somewhere in the same plausible band
    the real picker chooses from. If the regime-conditioned `pick_ell_star` can't beat this,
    the conditioning isn't buying anything. The upper bound is the regime's own observed
    max range, so random and the real picker face the same support (a fair control, not a
    strawman that always places absurdly wide limits).

    Parameters
    ----------
    epdf : Counter mapping ℓ (int, ticks) → count of observations.
    rng  : a seeded ``numpy.random.Generator`` (the caller owns reproducibility).

    Returns
    -------
    int : ℓ* in ticks, uniform in ``[0, max(support)]``. Returns 0 for an empty histogram.
    """
    if not epdf:
        return 0
    max_ell = int(max(epdf.keys()))
    if max_ell <= 0:
        return 0
    # randint upper bound is exclusive → +1 so max_ell itself is reachable.
    return int(rng.integers(0, max_ell + 1))


def chase_price(
    open_price: float,
    tick: float,
    *,
    policy: ChasePolicy,
    window_high: float | None = None,
    window_low: float | None = None,
    window_close: float | None = None,
) -> float:
    """Execution price when a static limit order goes UNFILLED over its τ-window.

    The fill side does not change *where* you get done when you finally cross the
    spread — only the sign of the resulting slippage, which the caller applies. So this
    returns a raw price, not a signed cost.

    Policies
    --------
    ``"close"`` : market-execute at the window's last close (the baseline chase).
    ``"mid"``   : market-execute at the window mid ``(H + L) / 2`` — on average a less
                  adverse fill than waiting for the close when price has trended away.

    `tick` is accepted for interface symmetry with the other models (mid/close need no
    rounding to the grid here; the backtest expresses the result in ticks downstream).
    """
    if policy == "close":
        if window_close is None:
            raise ValueError("policy='close' needs window_close")
        return float(window_close)
    if policy == "mid":
        if window_high is None or window_low is None:
            raise ValueError("policy='mid' needs window_high and window_low")
        return (float(window_high) + float(window_low)) / 2.0
    raise ValueError(f"unknown chase policy: {policy!r}")


def simulate_early_chase(
    side: Side,
    open_price: float,
    ell_star: int,
    tick: float,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    trigger_ticks: int,
) -> tuple[float, bool]:
    """Simulate a τ-window where a passive limit can fill OR we bail early once price
    runs ``trigger_ticks`` against us.

    No-lookahead: reads only the bars *inside* the decision window ``[t, t+τ]`` — ``ell_star``
    was chosen from data strictly before ``t``. Walking the realized window to settle the
    fill is outcome simulation, not signal construction (identical in spirit to the
    baseline reading the window close).

    Within a single bar we only have OHLC, not the tick path, so we resolve the
    **favorable limit first, then the adverse trigger** (the order's primary intent wins
    a same-bar tie). The bail is modelled as a stop executed at the trigger level.

    Sell (limit = open + ell_star ticks): per bar, in order —
      * ``high ≥ open + ell_star·tick``                       → fill at the limit.
      * else ``low ≤ open - trigger_ticks·tick``              → bail at ``open - trigger_ticks·tick``.
    Buy (limit = open - ell_star ticks): per bar —
      * ``low ≤ open - ell_star·tick``                        → fill at the limit.
      * else ``high ≥ open + trigger_ticks·tick``             → bail at ``open + trigger_ticks·tick``.
    Neither by window end → execute at the last close.

    Returns
    -------
    (price, filled) : execution price, and whether it filled at the passive limit.
    """
    if trigger_ticks <= 0:
        raise ValueError("trigger_ticks must be > 0")
    if len(highs) == 0 or not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs/lows/closes must be non-empty and equal length")

    if side == "sell":
        limit = open_price + ell_star * tick
        bail = open_price - trigger_ticks * tick
        for h, lo in zip(highs, lows, strict=True):
            if h >= limit:
                return limit, True
            if lo <= bail:
                return bail, False
    elif side == "buy":
        limit = open_price - ell_star * tick
        bail = open_price + trigger_ticks * tick
        for h, lo in zip(highs, lows, strict=True):
            if lo <= limit:
                return limit, True
            if h >= bail:
                return bail, False
    else:
        raise ValueError(f"unknown side: {side!r}")

    return float(closes[-1]), False
