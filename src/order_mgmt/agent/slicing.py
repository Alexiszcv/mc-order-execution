"""Order-slicing fill schemes — how to 'cut' a parent order WITHIN the τ window.

All schemes stay inside [t, t+τ) (the agent re-decides every 5 min, so we must not
stretch one order across windows it may later contradict). Each returns
(realized_price, fill_fraction) for a parent of unit size; the caller turns the
realized price into a shortfall vs the arrival price.

Reuses `backtest._simulate_fill` for the single-shot primitive so the fill model is
identical to the rest of the project (limit ℓ* ticks from the open; fills iff the
realized half-range reaches ℓ*, else chases at the close).

Schemes:
  - single     : one limit for the whole order over the window (the current behaviour).
  - time_slice : split into K equal children across sub-intervals; each child posts a
                 √-scaled limit and fills/chases independently → diversifies the tail.
  - blend      : fill (1−f) immediately at the open (certain), post f as a limit.
  - cutoff     : post the limit; if unfilled by a cutoff point inside the window, market
                 the remainder THEN (caps the worst-case chase vs waiting for the close).
"""

from __future__ import annotations

import math

import numpy as np

from order_mgmt.backtest import Side, _simulate_fill


def _ru_rd(open_j: float, high_j: float, low_j: float, tick: float) -> tuple[int, int]:
    return max(round((high_j - open_j) / tick), 0), max(round((open_j - low_j) / tick), 0)


def fill_single(
    side: Side, ell_star: int, o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray, tick: float
) -> tuple[float, float]:
    """One limit for the whole order over the window (current behaviour)."""
    open_j, close_j = float(o[0]), float(c[-1])
    ru, rd = _ru_rd(open_j, float(h.max()), float(low.min()), tick)
    price, filled, _ = _simulate_fill(side, ell_star, ru, rd, open_j, close_j, tick)
    return price, 1.0 if filled else 0.0


def fill_time_slice(
    side: Side, ell_star: int, o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray,
    tick: float, k: int = 3,
) -> tuple[float, float]:
    """Split the order into K equal children across sub-intervals of the window.

    Each child posts a limit scaled by √(1/K) (BM range scales with √time), and
    fills/chases independently against its own sub-interval. Parent price = mean of
    child prices (equal sizes); fill_fraction = filled children / K.
    """
    n = len(o)
    k = max(1, min(k, n))
    ell_sub = max(round(ell_star / math.sqrt(k)), 0)
    prices, n_filled = [], 0
    for g in np.array_split(np.arange(n), k):
        og, cg = float(o[g[0]]), float(c[g[-1]])
        ru, rd = _ru_rd(og, float(h[g].max()), float(low[g].min()), tick)
        price, filled, _ = _simulate_fill(side, ell_sub, ru, rd, og, cg, tick)
        prices.append(price)
        n_filled += int(filled)
    return float(np.mean(prices)), n_filled / k


def fill_blend(
    side: Side, ell_star: int, o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray,
    tick: float, f: float = 0.5,
) -> tuple[float, float]:
    """Fill (1−f) immediately at the open; post f as a limit over the window."""
    open_j, close_j = float(o[0]), float(c[-1])
    ru, rd = _ru_rd(open_j, float(h.max()), float(low.min()), tick)
    limit_price, filled, _ = _simulate_fill(side, ell_star, ru, rd, open_j, close_j, tick)
    realized = (1.0 - f) * open_j + f * limit_price
    return realized, (1.0 - f) + f * (1.0 if filled else 0.0)


def fill_cutoff(
    side: Side, ell_star: int, o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray,
    tick: float, cutoff_frac: float = 0.5,
) -> tuple[float, float]:
    """Post the limit; if unfilled by `cutoff_frac` of the window, market the remainder then.

    Walks the 1-min bars: the limit fills the first bar whose range reaches the level
    (buy: low ≤ open−ℓ*·tick; sell: high ≥ open+ℓ*·tick). If no bar fills by the cutoff
    index, execute at that bar's close — capping the chase at mid-window instead of the
    (often worse) window close.
    """
    n = len(o)
    open_j = float(o[0])
    cut = max(1, round(cutoff_frac * n))
    if side == "sell":
        level = open_j + ell_star * tick
        for i in range(n):
            if float(h[i]) >= level:
                return level, 1.0
            if i >= cut - 1:
                return float(c[i]), 0.0
    else:
        level = open_j - ell_star * tick
        for i in range(n):
            if float(low[i]) <= level:
                return level, 1.0
            if i >= cut - 1:
                return float(c[i]), 0.0
    return float(c[-1]), 0.0


def fill_capped(
    side: Side, ell_star: int, o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray,
    tick: float, cap: int = 8,
) -> tuple[float, float]:
    """Limit with a hard chase-cap (stop): keep the upside, truncate the tail.

    Post the limit at ℓ* (price improvement if it fills). But if price instead moves `cap`
    ticks the WRONG way from the open, market out immediately at that level — so the worst
    case is ≈ −cap ticks instead of riding an adverse move to the close. Asymmetric: the
    favorable fill is unbounded-good (ℓ*), the unfavorable outcome is bounded at the cap.
    """
    open_j = float(o[0])
    if side == "sell":
        limit, stop = open_j + ell_star * tick, open_j - cap * tick
        for i in range(len(o)):
            if float(h[i]) >= limit:
                return limit, 1.0  # filled at the better price
            if float(low[i]) <= stop:
                return stop, 0.0  # stopped out at the cap
    else:
        limit, stop = open_j - ell_star * tick, open_j + cap * tick
        for i in range(len(o)):
            if float(low[i]) <= limit:
                return limit, 1.0
            if float(h[i]) >= stop:
                return stop, 0.0
    return float(c[-1]), 0.0  # neither hit -> market at the close


# Registry: name -> (fill function, default kwargs). The script can override kwargs.
SCHEMES = {
    "single": (fill_single, {}),
    "time_slice": (fill_time_slice, {"k": 3}),
    "blend": (fill_blend, {"f": 0.5}),
    "cutoff": (fill_cutoff, {"cutoff_frac": 0.5}),
}
