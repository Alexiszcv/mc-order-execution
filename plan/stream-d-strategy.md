# Stream D — Strategy research / extension

- **Branch:** `stream/d-strategy`  ·  **Worktree:** `C:/Users/jltch/mc-wt/d`
- **Owns (edits):** `src/order_mgmt/strategy.py`, new `scripts/sweep_fill_rate.py` (+ any
  other new `scripts/sweep_*.py`), `tests/test_strategy.py`, new figures in
  `reports/figures/`.
- **Do NOT touch:** `backtest.py`, `baselines.py` (call them read-only — C owns them),
  `epdf.py`, `regime.py`, `app.py`.
- **Shared contract you own:** `pick_ell_star(epdf, fill_rate_target) -> int`. Backtest
  calls it. **New policies must be NEW functions** — do not change this signature.

## Goal
Push the strategy research forward: quantify the fill-rate / slippage trade-off, try
smarter chase policies, and broaden across markets — surfacing options with trade-offs for
the user, **not** picking a final parameter set (CLAUDE.md: that's the user's call).

## Why (context)
The strategy is currently one knob (`fill_rate_target`) and one policy (static limit ℓ*
ticks from the window open, chase at the window close if unfilled). The headline result is
**median +2…+7 ticks but mean ≈ 0** — the chase-on-unfill tail eats the gain. The open
research questions are: where is the fill/slippage sweet spot, and can a smarter chase
shrink the negative tail?

## Tasks
1. **Fill-rate sweep + Pareto curve.** New `scripts/sweep_fill_rate.py`: for
   `fill_rate_target ∈ {0.4, 0.5, 0.6, 0.7, 0.8}` (configurable), run `run_backtest_rolling`
   (v2 — the trustworthy one) per side on ≥2 markets, collect (achieved fill rate, mean
   slippage, median slippage). Plot the mean-slippage-vs-fill-rate Pareto curve and save to
   `reports/figures/pareto_fill_rate_<market>.png` (fixed seed where any RNG is involved).
   Print a table. This shows whether 0.6 is actually the sweet spot.
2. **Smarter chase policies** as NEW functions (don't break `pick_ell_star`). Candidates,
   each parameterised and documented:
   - *chase-at-mid*: on unfill, execute at the window's mid `(H+L)/2` instead of close.
   - *early-chase*: chase as soon as price moves > X ticks against the limit, not only at
     window close. (Needs intrabar logic in a small helper; keep no-lookahead — only bars
     up to the decision window are read.)
   Compare each against the baseline chase on the same v2 path; report tail shrinkage
   (e.g. 5th-percentile slippage) and fill rate.
3. **(Stretch) cost-aware objective.** Instead of a fixed fill-rate target, pick ℓ* that
   maximises an expected-cost objective `E[fill]·improvement − P[unfill]·chase_cost` over
   the per-regime ePDF. New function; surface as an alternative, with the trade-off vs the
   target-based picker.
4. **More markets.** Extend the sweep's market list to include Bunds (RX), EuroStoxx (VG),
   GBP (BP) — proves genericity. Reuse `pipeline.load_market_indexed` + `ticks.resolve_tick`.

## Tests to add (`tests/test_strategy.py` — extend)
- `pick_ell_star` *achieves* ~target: build a known integer ePDF, assert the survival at
  the returned ℓ* is ≥ target and that ℓ*+1 would drop below it (it's the largest such ℓ).
- Monotonicity: higher `fill_rate_target` ⇒ ℓ* non-increasing.
- Each new chase policy: a tiny deterministic scenario with a hand-computed expected price.

## Done criteria
- Pareto sweep script runs on ≥2 markets and saves figures (fixed seed).
- ≥1 alternative chase policy implemented + tested + compared to baseline in the script's
  output.
- Strategy tests added; `pytest -q` green; `ruff check .` clean.
- A short "options & trade-offs" summary appended to `notes/component-review.md` (or a new
  `notes/strategy-sweep.md`) for the user to choose from — **do not** hardcode a winner.
