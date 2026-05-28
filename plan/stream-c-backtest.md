# Stream C — Backtest correctness + efficiency

- **Branch:** `stream/c-backtest`  ·  **Worktree:** `C:/Users/jltch/mc-wt/c`
- **Owns (edits):** `src/order_mgmt/backtest.py`, `src/order_mgmt/baselines.py`, `tests/test_backtest.py`
- **Do NOT touch:** `strategy.py` (call it, don't edit — D owns it), `epdf.py`, `ranges.py`, `regime.py`, `app.py`.
- **Shared contract you own:** the keyword signatures of `run_backtest(...)` and
  `run_backtest_rolling(...)`. D, E, and `scripts/run_v1.py` call them. **Only ADD optional
  kwargs** (e.g. precomputed arrays); never rename or remove existing ones.

## Goal
Make v1 internally consistent (or precisely documented), kill the duplicated fill logic,
remove redundant recomputation, speed up VWAP, and guard the v1≈v2 headline with a test.

## Why (context)
From the cold review (`notes/component-review.md`):
1. **v1 state-assignment mismatch.** `build_epdf` *populates* regime cells using an
   expanding-prefix rank binning, but v1's decision loop *re-derives* the regime from
   **final full-history** sorted thresholds (`vol_sorted`/`range_sorted`/`dx_sorted`). A
   window can be accumulated into cell A but looked up in cell B. v1 is "permissive by
   design," but this is an *internal inconsistency*, and the published "v1≈v2 within 0.05
   ticks" result partly rests on it.
2. **Duplicated fill/slippage block** (~25 lines) verbatim across v1 and v2.
3. **Redundant compute.** Each of v1/v2 recomputes `compute_all_ranges` +
   `compute_ewma_series`; `run_v1.py` calls `compute_all_ranges` a 3rd time per side for
   the VWAP `t_list`. ~6 range passes per market.
4. **O(n²) VWAP** in `baselines.vwap_baseline` (boolean-mask `.loc` per window).
5. **Benchmark = window open** ⇒ TWAP ≡ 0 by construction; only VWAP is informative.

## Tasks
1. **Resolve the v1 mismatch.** Make v1 consistent: use the *same* state-assignment basis
   for both building the ePDF and looking it up. Simplest: have v1 assign states from the
   full-history sorted lists for *both* (its permissive intent) — i.e. rebuild `counts`
   with full-history thresholds, or assign at decision time from the same lists used to
   bin. Add a comment stating v1's invariant explicitly. If you'd rather not change v1's
   numbers, instead document the mismatch precisely in the docstring AND add the agreement
   test (task 5) so any drift is caught — but consistency is preferred.
2. **Factor `_simulate_fill(side, ell_star, ell_u_j, ell_d_j, open_j, close_j, tick) ->
   (price, filled: bool, slip)`** and call it from both v1 and v2. Refactor-only commit,
   separate from any behavior change.
3. **Compute once.** Add an *optional* path so callers can pass precomputed
   `(t_list, ell_r, ell_u, ell_d, vol_list, dx_list)` / EWMA arrays in via new keyword args
   (default `None` → compute internally, preserving the current signature). Update
   `scripts/run_v1.py` to compute ranges once per market and feed both v1, v2, and the VWAP
   `t_list`. (run_v1.py edit is allowed for C since it's the demo glue, not a shared API —
   but keep it minimal.)
4. **Speed up `vwap_baseline`** using the same groupby/searchsorted approach as
   `compute_all_ranges` (or fold VWAP into the single range pass). Behavior must be
   identical — assert via a small test.
5. **Add a v1↔v2 agreement test** on real Gold data: mean slippage agrees within a stated
   tolerance (e.g. 0.1 ticks) at τ=5, M=N=K=3, j_start=200. This guards the headline result
   in CI. Skip gracefully if Gold data absent (mirror existing `pytestmark`).
6. **Benchmark note:** add a docstring line clarifying TWAP-at-open is the zero baseline,
   so the "vs TWAP" framing isn't misread. (Switching to an arrival/VWAP benchmark is a
   research question — leave it for D unless trivial.)

## Done criteria
- v1 build/lookup use a consistent state basis (or the mismatch is precisely documented +
  guarded by the agreement test).
- Single `_simulate_fill` helper; no duplicated fill block.
- Optional precomputed-array path added (signatures backward-compatible); `run_v1.py`
  computes ranges once.
- `vwap_baseline` no longer O(n²); equivalence test passes.
- v1↔v2 agreement test added. `pytest -q` green, `ruff check .` clean. Runs on ≥2 markets.
