"""Sampling from empirical ePDFs (inverse-transform & composition).

A regime's ePDF is a `collections.Counter` mapping a tick value ℓ → count, i.e. an
empirical PMF. `sample_from_epdf` is the cheat-sheet inverse-transform method on its
CDF; `sample_composition` is the composition method (draw a regime, then draw a value
from it).
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def build_cdf(epdf: Counter) -> tuple[np.ndarray, np.ndarray]:
    """Return (support, cumprob) for a Counter ePDF, sorted ascending by tick value.

    `support[i]` is a tick value; `cumprob[i] = P(R <= support[i])`. The last entry is
    forced to exactly 1.0 so inverse-transform draws can never fall off the end.
    """
    if not epdf:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    support = np.array(sorted(epdf.keys()), dtype=np.int64)
    counts = np.array([epdf[int(k)] for k in support], dtype=float)
    total = counts.sum()
    cumprob = np.cumsum(counts) / total
    cumprob[-1] = 1.0  # guard against float drift below 1
    return support, cumprob


def sample_from_epdf(epdf: Counter, n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Inverse-transform: draw `n` tick values from the empirical CDF of `epdf`.

    For U ~ U(0,1), return the smallest ℓ with F̃(ℓ) >= U (cheat sheet: x = F^{-1}(U)).
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    support, cumprob = build_cdf(epdf)
    if support.size == 0:
        raise ValueError("cannot sample from an empty ePDF")
    u = rng.random(n)
    idx = np.searchsorted(cumprob, u, side="left")
    idx = np.clip(idx, 0, support.size - 1)
    return support[idx]


def sample_composition(
    counts_RU: dict,
    counts_RD: dict,
    n: int,
    *,
    cell_weights: dict[tuple[int, int, int], float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Composition method: draw a regime cell ~ `cell_weights`, then (R_U, R_D) from it.

    R_U and R_D are drawn independently within a cell (the strategy only ever uses one
    side per decision, so their within-window dependence is irrelevant here). Cells with
    an empty R_U *or* R_D counter are skipped from the weighting.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    cells = [
        c
        for c, w in cell_weights.items()
        if w > 0 and counts_RU.get(c) and counts_RD.get(c)
    ]
    if not cells:
        raise ValueError("no populated cells to sample from")
    w = np.array([cell_weights[c] for c in cells], dtype=float)
    w /= w.sum()

    chosen = rng.choice(len(cells), size=n, p=w)
    r_u = np.empty(n, dtype=np.int64)
    r_d = np.empty(n, dtype=np.int64)
    for ci, cell in enumerate(cells):
        mask = chosen == ci
        k = int(mask.sum())
        if k == 0:
            continue
        r_u[mask] = sample_from_epdf(counts_RU[cell], k, rng=rng)
        r_d[mask] = sample_from_epdf(counts_RD[cell], k, rng=rng)
    return r_u, r_d
