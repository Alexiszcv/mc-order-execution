# Stream B — Data / liquidity unification

- **Branch:** `stream/b-data`  ·  **Worktree:** `C:/Users/jltch/mc-wt/b`
- **Owns (edits):** `src/order_mgmt/loader.py`, `src/plot_volume.py`, `src/order_mgmt/ticks.py`, `tests/test_loader.py`
- **Do NOT touch:** `app.py`, `backtest.py`, `regime.py`, `strategy.py`.
- **Shared contract you own:** `plot_volume._compute_stats(df) -> (daily, tick,
  proper_days, n_green, n_total)`. C and E consume it. If you change the return shape, say
  so in your merge PR so they rebase.

## Goal
One liquidity definition, one tick-inference path, and tests covering the roll and the
filter — the data foundation the other streams build on.

## Why (context)
Three concrete problems found in the cold review (`notes/component-review.md`):
1. **Two conflicting "active day" filters.** `loader.drop_low_liquidity_days` keeps days
   with `≥ 0.90 × EXPECTED_MIN_PER_DAY` where `EXPECTED_MIN_PER_DAY = 480` (8h).
   `plot_volume._compute_stats` keeps days with `≥ 0.90 × observed-max-traded-mins`. The
   datasets are ~23h futures (Gold, Bunds, …), so the absolute 480 is too lenient (thin
   days pass) and the two definitions disagree on which days survive.
2. **Duplicated tick inference.** `_compute_stats` (min positive price diff + magnitude
   round) is copy-pasted in `get_tick`. Same logic, two places.
3. **No test** pins the roll crossover (`pick_active_contract`) or the liquidity filter.

## Tasks
1. **Unify the liquidity definition.** Recommended: make the *relative* definition
   (`≥ fraction × observed-max-traded-mins per contract/session`) the single source of
   truth, since it auto-adapts to each market's session length. Refactor
   `loader.drop_low_liquidity_days` and `_compute_stats` to call one shared helper. If you
   keep an absolute bound, derive `EXPECTED_MIN_PER_DAY` from the data (e.g. modal daily
   bar count) rather than hardcoding 480. **Surface the choice in the PR** — do not silently
   change which days are kept for downstream backtests without noting the before/after day
   counts on Gold + one other market.
2. **Dedup tick inference** into one helper (e.g. `infer_tick(prices) -> float`) used by
   both `_compute_stats` and `get_tick`. Keep `get_tick(csv_path)` as a thin wrapper.
3. **Cross-check `ticks.py:TICK_TABLE`** against the spec PDF; correct any wrong value and
   cite the source line in a comment.
4. **Remove the broken `plot_volume.plot_volume()`** (it calls `_build_figure(csv_path)`
   with the wrong signature — would crash). It's dead. (This is a cleanup folded into B.)

## Tests to add (`tests/test_loader.py` — extend)
- `pick_active_contract` roll: synth two contracts whose daily volume crosses over on a
  known date; assert the winner flips exactly there and bars from the loser are dropped on
  each side.
- Liquidity filter: synth days at 100%, 91%, 89% of the session length; assert the
  unified filter keeps/drops the right ones at `fraction=0.90`.
- `infer_tick`: known price grid (e.g. 0.10-spaced) → 0.10; a 0.0001 grid → 0.0001.

## Done criteria
- One liquidity definition, called from both loader and `_compute_stats`.
- One tick-inference helper.
- New roll + liquidity + tick tests pass; `pytest -q` green; `ruff check .` clean.
- Loader runs end-to-end on **Gold + one other market**; PR notes the active-day counts
  before/after the filter change for both.
