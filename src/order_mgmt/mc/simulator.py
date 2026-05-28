"""Phase B — forward Monte Carlo of the limit-order strategy.

Picks ℓ* from the *historical empirical* decision ePDF (so the decision rule is held
fixed), then draws the realized rangeUp/rangeDn from the chosen `range_model`
("empirical" | "fitted" | "gbm") and scores fill rate + slippage over `n_paths`.

NO-LOOKAHEAD: this layer consumes pre-built ePDFs / fits / σ̄ (built only from windows
< j upstream) and adds no time ordering of its own — output is a pure function of inputs +
seed. The "bridge" model needs the realized close (look-ahead) and is rejected here; it
lives in `validation.py`.

Slippage convention (shared across models): a fill executes ℓ* ticks better than the open
(+ℓ*); a non-fill chases at the close, whose offset from the open is the BM terminal in
ticks — coupled to the path for "gbm", drawn from N(0, σ_window²) for "empirical"/"fitted".
Slippage agreement across models is therefore looser than fill-rate agreement.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from order_mgmt.strategy import pick_ell_star

from .paths import simulate_gbm_paths
from .results import MCEstimate, MCRunResult, Side
from .samplers import sample_from_epdf


def _fill_and_slippage(realized: np.ndarray, close_ticks: np.ndarray, side: Side, ell_star: int):
    """Per-path fill indicator (float 0/1) and signed slippage in ticks."""
    fill = realized >= ell_star
    chase = close_ticks if side == "sell" else -close_ticks
    slip = np.where(fill, float(ell_star), chase)
    return fill.astype(float), slip


def _simulate_cell(
    decision_epdf: Counter,
    *,
    side: Side,
    fill_rate_target: float,
    n_paths: int,
    range_model: str,
    tau: int,
    sigma_bar: float,
    outcome_epdf: Counter | None,
    fitted,
    n_sub: int,
    rng: np.random.Generator,
):
    """Return (ell_star, fill_array, slippage_array) for one regime under one model."""
    ell_star = pick_ell_star(decision_epdf, fill_rate_target)
    sigma_window = sigma_bar * math.sqrt(tau)

    if range_model == "gbm":
        r_u, r_d, close = simulate_gbm_paths(
            sigma_bar, tau, n_paths, n_sub=n_sub, round_ticks=True, rng=rng
        )
        realized = r_u if side == "sell" else r_d
    elif range_model == "empirical":
        epdf = outcome_epdf if outcome_epdf is not None else decision_epdf
        realized = sample_from_epdf(epdf, n_paths, rng=rng)
        close = rng.standard_normal(n_paths) * sigma_window
    elif range_model == "fitted":
        if fitted is None:
            raise ValueError("range_model='fitted' requires a FitResult via `fitted=`")
        realized = fitted.sample(n_paths, rng=rng)
        close = rng.standard_normal(n_paths) * sigma_window
    elif range_model == "bridge":
        raise ValueError(
            "range_model='bridge' needs the realized close (look-ahead); "
            "use validation, not run_mc_execution"
        )
    else:
        raise ValueError(f"unknown range_model {range_model!r}")

    fill, slip = _fill_and_slippage(realized, close, side, ell_star)
    return ell_star, fill, slip


def run_mc_execution(
    decision_epdf: Counter,
    *,
    side: Side,
    fill_rate_target: float,
    n_paths: int,
    range_model: str,
    tau: int,
    sigma_bar: float,
    outcome_epdf: Counter | None = None,
    fitted=None,
    n_sub: int = 20,
    ci: float = 0.95,
    seed: int = 0,
) -> MCRunResult:
    """Forward-simulate one regime under one range model -> fill rate + slippage distribution."""
    rng = np.random.default_rng(seed)
    ell_star, fill, slip = _simulate_cell(
        decision_epdf,
        side=side,
        fill_rate_target=fill_rate_target,
        n_paths=n_paths,
        range_model=range_model,
        tau=tau,
        sigma_bar=sigma_bar,
        outcome_epdf=outcome_epdf,
        fitted=fitted,
        n_sub=n_sub,
        rng=rng,
    )
    return MCRunResult(
        range_model=range_model,
        side=side,
        ell_star=ell_star,
        fill_rate=MCEstimate.from_samples(fill, ci=ci, method="clt"),
        avg_slippage_ticks=MCEstimate.from_samples(slip, ci=ci, method="clt"),
        slippage_samples=slip.tolist(),
        n_paths=int(fill.size),
        seed=seed,
    )


def run_marginal_mc(
    counts_decision: dict,
    *,
    side: Side,
    fill_rate_target: float,
    n_paths: int,
    range_model: str,
    tau: int,
    sigma_by_cell: dict,
    cell_weights: dict,
    fits: dict | None = None,
    n_sub: int = 20,
    ci: float = 0.95,
    seed: int = 0,
) -> MCRunResult:
    """Regime-marginal MC: mix per-cell simulations by `cell_weights` (apples-to-apples
    with the whole-history backtest). `counts_decision` holds the side's ePDF per cell."""
    rng = np.random.default_rng(seed)
    cells = [
        c
        for c, w in cell_weights.items()
        if w > 0
        and counts_decision.get(c)
        and c in sigma_by_cell
        and (range_model != "fitted" or (fits or {}).get(c) is not None)
    ]
    if not cells:
        raise ValueError("no usable regime cells for marginal MC")
    total_w = sum(cell_weights[c] for c in cells)

    fills, slips = [], []
    for c in cells:
        n_c = round(n_paths * cell_weights[c] / total_w)
        if n_c <= 0:
            continue
        _ell, fill, slip = _simulate_cell(
            counts_decision[c],
            side=side,
            fill_rate_target=fill_rate_target,
            n_paths=n_c,
            range_model=range_model,
            tau=tau,
            sigma_bar=sigma_by_cell[c],
            outcome_epdf=None,
            fitted=(fits or {}).get(c),
            n_sub=n_sub,
            rng=rng,
        )
        fills.append(fill)
        slips.append(slip)

    fill = np.concatenate(fills)
    slip = np.concatenate(slips)
    return MCRunResult(
        range_model=range_model,
        side=side,
        ell_star=-1,  # mixed across cells
        fill_rate=MCEstimate.from_samples(fill, ci=ci, method="clt"),
        avg_slippage_ticks=MCEstimate.from_samples(slip, ci=ci, method="clt"),
        slippage_samples=slip.tolist(),
        n_paths=int(fill.size),
        seed=seed,
    )


def stress_sigma(
    decision_epdf: Counter,
    *,
    side: Side,
    fill_rate_target: float,
    n_paths: int,
    tau: int,
    sigma_bar: float,
    scales,
    n_sub: int = 20,
    seed: int = 0,
) -> list[MCRunResult]:
    """Re-run the forward-GBM model with σ̄·scale for each scale — a volatility stress test."""
    out = []
    for i, s in enumerate(scales):
        out.append(
            run_mc_execution(
                decision_epdf,
                side=side,
                fill_rate_target=fill_rate_target,
                n_paths=n_paths,
                range_model="gbm",
                tau=tau,
                sigma_bar=sigma_bar * float(s),
                n_sub=n_sub,
                seed=seed + i,
            )
        )
    return out
