# Stream E — UI alignment

- **Branch:** `stream/e-ui`  ·  **Worktree:** `C:/Users/jltch/mc-wt/e`
- **Owns (edits):** `src/app.py`, `src/plotting.py`
- **Do NOT touch:** `backtest.py`, `strategy.py`, `loader.py`, `plot_volume.py`,
  `pipeline.py` (call them read-only). If you find you *need* a change in one of those,
  stop and flag it as a cross-stream coordination point rather than editing it.
- **Consumes (must not break):** `_compute_stats` (B owns), `run_backtest*` (C owns),
  `load_market_indexed` (pipeline). If those signatures move under you, rebase.

## Goal
Make the parameter-tuning UI (the `app.py` localhost explorer — a final-submission
deliverable) consistent with the script pipeline and show the trustworthy numbers.

## Why (context)
Three inconsistencies from the cold review (`notes/component-review.md`):
1. **UI ≠ script data foundation.** `app.py` loads a *single* CSV via `_load_1min` and does
   no contract roll, while `scripts/run_v1.py` uses the roll-aware multi-contract loader.
   So the UI analyses one expiry; the script analyses the rolled series. Same "analysis,"
   different data.
2. **UI shows v1, not v2.** Given v1's documented state-assignment caveat (Stream C), the
   UI is currently surfacing the *less* trustworthy backtest.
3. **No `fill_rate_target` slider** — the strategy's single most important knob isn't
   exposed, even though the UI's whole point is parameter tuning.

## Tasks
1. **Roll-aware data in the UI.** Add a mode (or switch the default) so a *market* can be
   selected and loaded via `pipeline.load_market_indexed`, not just a single contract CSV.
   Keep single-contract view available if useful, but make rolled-market the headline path
   so the UI matches `run_v1.py`. Resolve tick via `ticks.resolve_tick` on the first rolled
   contract stem (as `run_v1.py` does).
2. **Show v2 (no-lookahead).** Call `run_backtest_rolling` for the backtest panel (or show
   v1 and v2 side by side and label which is no-lookahead). Update the panel title/caption.
3. **Add a `fill_rate_target` slider** (range ~0.3–0.9, default 0.6), wired through the
   query params like the existing τ / half-life / M / N / K sliders, and pass it into the
   backtest call.
4. **Dedup the load path.** `app.py`, `epdf._load_1min`, and `plot_volume` each have a CSV
   loader. Have the UI use a single loading entry point (the pipeline for markets; one
   shared `_load_1min` for the single-contract path) rather than its own copy. Do this
   without editing `plot_volume.py`/`loader.py` (B owns them) — if a shared helper must
   move, flag it.
5. Keep `plotting.py` (histogram figure builder) tidy; adjust only if the new panels need
   it.

## Manual verification (UI changes — required, per CLAUDE.md)
- Run `python src/app.py`, open `http://localhost:8000`, select a market, move each slider
  (τ, half-life, M, N, K, fill_rate_target) and confirm panels re-render without error.
- Confirm the backtest panel reports v2 numbers and the fill-rate slider visibly changes
  fill rate / slippage. Note in the PR that you exercised it in the browser (type checks
  alone are not sufficient for UI).

## Done criteria
- UI loads rolled markets and matches `run_v1.py`'s data foundation.
- Backtest panel shows v2; fill-rate slider works end-to-end.
- No duplicated CSV-load logic introduced; no edits to B/C-owned files.
- `ruff check .` clean; existing `pytest -q` still green (UI is not unit-tested, but must
  not break imports). Browser-verified as above.
