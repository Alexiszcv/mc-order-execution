"""Validation throughline — does the Monte Carlo agree with reality / across models?

`compare_mc_vs_backtest` cross-checks a (regime-marginal) MC run against the historical
`run_backtest_rolling` result; fill rate is the robust quantity and the only one that
raises a divergence flag (slippage agreement is looser by construction — see `simulator`).
`compare_models` lays the three range models side by side against the empirical benchmark.
"""

from __future__ import annotations

from .results import MCRunResult


def compare_mc_vs_backtest(
    mc: MCRunResult,
    bt,
    *,
    tol_fill: float = 0.05,
    tol_slip: float = 1.0,
) -> dict:
    """Compare an MC run to a `BacktestResult`. Flags only fill-rate divergence."""
    bt_fill = float(bt.fill_rate)
    bt_slip = float(bt.avg_slippage_ticks)
    mc_fill = mc.fill_rate.mean
    mc_slip = mc.avg_slippage_ticks.mean

    fill_in_ci = mc.fill_rate.ci_low <= bt_fill <= mc.fill_rate.ci_high
    fill_within = abs(mc_fill - bt_fill) <= tol_fill or fill_in_ci
    slip_within = abs(mc_slip - bt_slip) <= tol_slip

    flag = None
    if not fill_within:
        flag = (
            f"fill-rate divergence: MC {mc_fill:.3f} (CI "
            f"[{mc.fill_rate.ci_low:.3f}, {mc.fill_rate.ci_high:.3f}]) "
            f"vs backtest {bt_fill:.3f}"
        )
    return {
        "fill_within_ci": fill_within,
        "slip_within_ci": slip_within,
        "flag": flag,
        "mc_fill": mc_fill,
        "bt_fill": bt_fill,
        "mc_slip": mc_slip,
        "bt_slip": bt_slip,
    }


def compare_models(
    results: dict,
    *,
    fit_summary: dict | None = None,
) -> dict:
    """Side-by-side of range models vs the empirical benchmark.

    `results` maps model name ("empirical" | "fitted" | "gbm") -> MCRunResult. Differences
    are reported relative to "empirical" (the yardstick). `fit_summary` optionally carries
    goodness-of-fit info (e.g. mean AIC / KS / selected-family counts).
    """
    base = results.get("empirical")
    rows = {}
    for name, r in results.items():
        row = {
            "fill_rate": r.fill_rate.mean,
            "fill_ci": (r.fill_rate.ci_low, r.fill_rate.ci_high),
            "avg_slippage_ticks": r.avg_slippage_ticks.mean,
        }
        if base is not None and name != "empirical":
            row["fill_diff_vs_empirical"] = r.fill_rate.mean - base.fill_rate.mean
            row["slip_diff_vs_empirical"] = (
                r.avg_slippage_ticks.mean - base.avg_slippage_ticks.mean
            )
        rows[name] = row
    out = {"models": rows}
    if fit_summary is not None:
        out["fit_summary"] = fit_summary
    return out
