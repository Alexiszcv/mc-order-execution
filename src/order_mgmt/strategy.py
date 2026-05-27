"""Order-placement strategy: pick the limit-order distance from an empirical ePDF."""

from __future__ import annotations

from collections import Counter


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
