"""End-to-end Monte Carlo smoke test on >=2 markets (Stream F deliverable).

For each market: roll-aware load -> ranges/EWMA/ePDF -> per-regime fits -> historical
no-lookahead backtest (v2) -> regime-marginal MC under the empirical, fitted, and gbm range
models. Asserts the empirical-MC fill rate agrees with the backtest (no divergence flag) and
prints the three-way model comparison + goodness-of-fit summary. Confirms the generic-module
requirement across tick sizes.

Run: python scripts/run_mc_smoke.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epdf import build_epdf  # noqa: E402
from order_mgmt.backtest import run_backtest_rolling  # noqa: E402
from order_mgmt.mc.fit import fit_all_regimes  # noqa: E402
from order_mgmt.mc.paths import sigma_from_range_level  # noqa: E402
from order_mgmt.mc.simulator import run_marginal_mc  # noqa: E402
from order_mgmt.mc.validation import compare_mc_vs_backtest, compare_models  # noqa: E402
from order_mgmt.pipeline import load_market_indexed  # noqa: E402
from order_mgmt.ticks import resolve_tick  # noqa: E402
from plot_volume import _compute_stats  # noqa: E402
from ranges import compute_all_ranges  # noqa: E402
from regime import compute_ewma_series  # noqa: E402

TAU = 5
HALF_LIFE = 20
M = N = K = 3
J_START = 200
FILL_RATE_TARGET = 0.6
N_PATHS = 40_000
SIDE = "sell"  # sells read the R_U distribution

MARKETS = ["Gold", "Nasdaq", "EuroStoxx"]  # try in order; need >=2 present


def _find_market_dir(substring: str) -> Path | None:
    data = ROOT / "data"
    if not data.exists():
        return None
    for p in sorted(data.iterdir()):
        if p.is_dir() and substring.lower() in p.name.lower():
            return p
    return None


def _cell_mean(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum(ell * c for ell, c in counter.items()) / total


def run_market(substring: str) -> dict | None:
    market_dir = _find_market_dir(substring)
    if market_dir is None:
        print(f"[{substring}] data dir not found - skipping")
        return None
    df = load_market_indexed(market_dir)
    if df.empty:
        print(f"[{substring}] no usable bars - skipping")
        return None

    first_stem = df["contract"].iloc[0]
    df_ohlcv = df[["open", "high", "low", "close", "volume"]]
    _, inferred_tick, proper_days, _, _ = _compute_stats(df_ohlcv)
    tick = resolve_tick(first_stem, inferred_tick)

    t_list, ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(
        df_ohlcv, TAU, tick, proper_days
    )
    ewma_range, ewma_vol = compute_ewma_series(t_list, ell_r, vol_list, HALF_LIFE)
    counts_RU, counts_RD, _thr = build_epdf(
        t_list, ell_u, ell_d, list(ewma_vol), list(ewma_range), dx_list,
        M=M, N=N, K=K, j_start=J_START,
    )

    decision = counts_RU  # sell side
    cell_weights = {c: sum(decision[c].values()) for c in decision}
    sigma_by_cell = {
        c: sigma_from_range_level(_cell_mean(counts_RU[c]) + _cell_mean(counts_RD[c]), TAU)
        for c in decision
    }
    fits = fit_all_regimes(decision)

    bt = run_backtest_rolling(
        df_ohlcv, tau=TAU, tick=tick, proper_days=proper_days, side=SIDE,
        fill_rate_target=FILL_RATE_TARGET, half_life=HALF_LIFE,
        M=M, N=N, K=K, j_start=J_START,
    )

    mc = {}
    for model in ("empirical", "fitted", "gbm"):
        mc[model] = run_marginal_mc(
            decision, side=SIDE, fill_rate_target=FILL_RATE_TARGET, n_paths=N_PATHS,
            range_model=model, tau=TAU, sigma_by_cell=sigma_by_cell,
            cell_weights=cell_weights, fits=fits, seed=0,
        )

    cmp = compare_mc_vs_backtest(mc["empirical"], bt, tol_fill=0.10)
    fit_summary = {
        "n_cells_fit": len(fits),
        "family_counts": dict(Counter(f.family for f in fits.values())),
        "mean_ks": float(np.mean([f.ks_stat for f in fits.values()])) if fits else float("nan"),
    }
    models = compare_models(mc, fit_summary=fit_summary)

    print(f"\n=== {market_dir.name}  (tick={tick:g}, {len(decision)} regimes, {bt.n_decisions} decisions) ===")
    print(f"  backtest v2: fill={bt.fill_rate:.1%}  avg_slip={bt.avg_slippage_ticks:+.2f}t")
    for model, r in mc.items():
        print(
            f"  MC {model:9}: fill={r.fill_rate.mean:.1%} "
            f"CI[{r.fill_rate.ci_low:.1%},{r.fill_rate.ci_high:.1%}]  "
            f"avg_slip={r.avg_slippage_ticks.mean:+.2f}t"
        )
    print(f"  fits: {fit_summary['family_counts']}  mean KS={fit_summary['mean_ks']:.3f}")
    print(f"  validation (empirical MC vs backtest): flag={cmp['flag']}")

    return {"market": market_dir.name, "cmp": cmp, "models": models, "bt": bt, "mc": mc}


def main() -> None:
    results = []
    for sub in MARKETS:
        res = run_market(sub)
        if res is not None:
            results.append(res)
        if len(results) >= 2:
            break

    assert len(results) >= 2, "need data for at least two markets"
    for res in results:
        assert res["cmp"]["flag"] is None, (
            f"{res['market']}: MC/backtest fill-rate divergence: {res['cmp']['flag']}"
        )
    print(f"\nOK - MC layer ran end-to-end on {len(results)} markets; no fill-rate divergence.")


if __name__ == "__main__":
    main()
