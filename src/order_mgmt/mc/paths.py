"""Phase B — Brownian-motion price-path models for the per-window range.

Pure math, no data dependency. The intra-window price is a driftless Brownian motion in
tick space started at the open (0); rangeUp/rangeDn are the path's running max / −min. Over
a ~5-min window drift is negligible, so arithmetic and geometric BM nearly coincide
(arithmetic is the default). The reflection principle gives the closed-form check
`halfnormal_range_mean`. The Brownian-bridge variant additionally pins the close — it uses
look-ahead and is for validation only (see `validation.py`).

Calibration: per-bar vol σ̄ is backed out of the running range *level* E[R] via the BM range
relation E[R] = σ_window·√(8/π) with σ_window = σ̄·√τ.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT_8_OVER_PI = math.sqrt(8.0 / math.pi)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def sigma_from_range_level(ewma_range_ticks: float, tau: int) -> float:
    """Per-bar BM vol σ̄ (ticks) implied by the running range level E[R] over τ bars.

    Inverts E[R] = σ̄·√τ·√(8/π). Returns 0 for a non-positive range level.
    """
    if ewma_range_ticks <= 0 or tau <= 0:
        return 0.0
    return float(ewma_range_ticks / (math.sqrt(tau) * _SQRT_8_OVER_PI))


def halfnormal_range_mean(sigma_bar: float, tau: int) -> float:
    """Closed-form E[rangeUp] for driftless BM: σ_window·√(2/π), σ_window = σ̄·√τ.

    (The full-range mean is √(8/π)·σ_window = 2× this, by symmetry.)
    """
    return float(sigma_bar * math.sqrt(tau) * _SQRT_2_OVER_PI)


def _increments(sigma_bar: float, tau: int, n_paths: int, n_sub: int, rng):
    """(n_paths, n_steps) BM path in tick space; per-step var set so total var = τ·σ̄²."""
    n_steps = tau * n_sub
    sigma_step = sigma_bar / math.sqrt(n_sub)
    z = rng.standard_normal((n_paths, n_steps))
    return np.cumsum(sigma_step * z, axis=1)


def _ranges_from_path(path: np.ndarray, round_ticks: bool):
    """rangeUp = max(0, running max), rangeDn = −min(0, running min); open = 0."""
    hi = np.maximum(path.max(axis=1), 0.0)
    lo = np.minimum(path.min(axis=1), 0.0)
    r_u = hi
    r_d = -lo
    if round_ticks:
        r_u = np.maximum(np.round(r_u), 0).astype(np.int64)
        r_d = np.maximum(np.round(r_d), 0).astype(np.int64)
    return r_u, r_d


def simulate_gbm_ranges(
    sigma_bar: float,
    tau: int,
    n_paths: int,
    *,
    n_sub: int = 1,
    geometric: bool = False,
    s0_ticks: float | None = None,
    round_ticks: bool = True,
    rng: np.random.Generator,
):
    """Forward BM over τ bars (only the open is known); return (rangeUp, rangeDn) in ticks.

    `n_sub` sub-steps per bar refine the path toward the continuous max (a coarse path
    under-estimates the true max — the discrete-vs-continuous bias). `geometric=True`
    multiplies the path onto `s0_ticks` (required) instead of adding; over short windows the
    two nearly coincide. `round_ticks=False` returns the raw continuous ranges (used to
    check the reflection-principle formula).
    """
    if n_paths <= 0:
        empty = np.empty(0)
        return empty, empty
    path = _increments(sigma_bar, tau, n_paths, n_sub, rng)
    if geometric:
        if s0_ticks is None or s0_ticks <= 0:
            raise ValueError("geometric=True requires a positive s0_ticks")
        price = s0_ticks * np.exp(path / s0_ticks)  # log-return path scaled to ticks
        path = price - s0_ticks
    return _ranges_from_path(path, round_ticks)


def simulate_gbm_paths(
    sigma_bar: float,
    tau: int,
    n_paths: int,
    *,
    n_sub: int = 1,
    round_ticks: bool = True,
    rng: np.random.Generator,
):
    """Like `simulate_gbm_ranges` but also returns the terminal (close − open) in ticks.

    The terminal feeds the chase-on-unfill slippage in the execution simulator, coupled to
    the same path that produced the range.
    """
    if n_paths <= 0:
        empty = np.empty(0)
        return empty, empty, empty
    path = _increments(sigma_bar, tau, n_paths, n_sub, rng)
    r_u, r_d = _ranges_from_path(path, round_ticks)
    return r_u, r_d, path[:, -1]


def simulate_bridge_ranges(
    sigma_bar: float,
    tau: int,
    dx_close_ticks,
    *,
    n_sub: int = 1,
    round_ticks: bool = True,
    rng: np.random.Generator,
):
    """Brownian bridge pinned at open=0 and close=`dx_close_ticks[i]` per path.

    LOOK-AHEAD: conditions on the realized close. For validation/decomposition only — never
    in the decision path. Returns (rangeUp, rangeDn).
    """
    b = np.asarray(dx_close_ticks, dtype=float)
    n_paths = b.size
    if n_paths == 0:
        empty = np.empty(0)
        return empty, empty
    w = _increments(sigma_bar, tau, n_paths, n_sub, rng)
    n_steps = w.shape[1]
    t = np.arange(1, n_steps + 1, dtype=float) / n_steps  # t/T at each step
    # B_t = W_t - (t/T)(W_T - b); pins B_T = b, B_0 = 0
    bridge = w - t[None, :] * (w[:, -1:] - b[:, None])
    return _ranges_from_path(bridge, round_ticks)
