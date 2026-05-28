# Component review (2026-05-28)

A cold read of every `src/` module + tests, to ground the planning work. This is the
shared reference for the parallel work plans in `plan/`. Findings are tagged
**[correctness] / [consistency] / [efficiency] / [extension] / [test-gap]** and
prioritized at the end.

## Module map — two layers, one seam

The repo has **two coexisting module systems**, a merge artifact from folding the
team's code into the personal branch:

| Layer | Files | Style | Import form |
|---|---|---|---|
| `order_mgmt/` package | `loader.py`, `ticks.py`, `strategy.py`, `backtest.py`, `baselines.py`, `pipeline.py` | typed, dataclasses, docstrings, `MarketSpec` | `from order_mgmt.x import …` |
| flat `src/` ("team") | `ranges.py`, `regime.py`, `epdf.py`, `plotting.py`, `plot_volume.py`, `app.py` | positional args, loose types, French→English residue | `from ranges import …` (pythonpath=["src"]) |

`backtest.py` and `app.py` straddle the seam — they import from both. This is **the**
structural decision the project hasn't made: unify into the package, or keep flat for
team PR-compatibility. Everything downstream (imports, tests, packaging) hinges on it.

## Per-component

### loader.py — roll-aware multi-contract loader
- `load_market` globs `*.csv` (skips `AIAgent_*`), loads each contract, calls
  `pick_active_contract` (per-date highest-volume contract wins → implicit roll), then
  `drop_low_liquidity_days`.
- `pick_active_contract`: concat all, group by (date, contract), `idxmax` daily volume,
  keep winner per date. Clean and genuinely generic. Good.
- **[correctness] `EXPECTED_MIN_PER_DAY = 480`** (8h). These are ~23h futures (Gold,
  Bunds, etc.). 90%×480 = 432 min is a *very* low bar — thin days with 500–600 of an
  expected ~1380 min pass. The absolute threshold likely under-filters. See the
  conflicting definition in `_compute_stats` below.

### ticks.py — spec tick-size table
- `TICK_TABLE` + `lookup_tick` (alpha-prefix match, longest-key-first) + `resolve_tick`
  (spec value if known, else fallback). This is the right pattern. Good.
- Minor: verify every table value against the PDF (GC=0.10, ES/NQ=0.25, BP, JY, RX, VG,
  HO) — these are cited in the docstring but worth a spec cross-check.

### ranges.py — R / R_U / R_D per τ-window
- Two implementations: `compute_ranges` (readable, pandas-sliced, per-day loop) and
  `compute_all_ranges` (fast: groupby once, int64-ns + `searchsorted`). Only the fast
  one is used downstream.
- Windows are **non-overlapping** `[t, t+τ)` from the day's first bar. Validity = exactly
  τ bars and no midnight cross. Correct per spec.
- `R_U`/`R_D` use `max(…, 0.0)` clamp; `R` from hi−lo. Identity `R = R_U + R_D` holds to
  ±1 tick (rounding). Tested.
- `dx[i] = round((open[i+1]−open[i])/tick)`, NaN when windows i,i+1 not strictly
  adjacent. Correct.
- **[correctness][subtle] the pandas-3.0 fix** (`.as_unit("ns").asi8`) is load-bearing:
  on pandas 3.0 the default datetime unit is µs, so raw `asi8` returned µs and the
  `// NS_PER_DAY` midnight check silently broke. Keep; PR back to team.
- **[consistency]** `compute_ranges` (the slow twin) is dead weight if only the fast one
  is used — either delete or keep strictly as the brute-force test oracle (it isn't
  currently used as one).

### regime.py — EWMA/EWMV + regime figure
- `ewma_ewmv(eta, half_life)`: λ = 2^(−1/m). Init at j=2 with eta[0]; loop j=3..n uses
  eta[j−1].
- **[correctness] THE eta[1] skip.** At j=2 it loads eta[0]; the loop starts at j=3 and
  reads eta[2]. **eta[1] (the 2nd observation) is never folded in.** Spec Algorithm 1
  folds in every observation η_{j−1}. This is an off-by-one — the loop body should
  consume eta[1] at the first iteration. Effect is small (one dropped sample, washed out
  by the EWMA decay) but it's a real deviation from spec.
- **[test-gap] the EWMA test enshrines the bug.** `test_ewma.py::_brute_force` mirrors
  the *source loop* (also skips eta[1]), so the test only proves "code matches its own
  docstring," not "code matches the spec." A spec-derived oracle would fail. This test
  gives false confidence.
- EWMV uses the *current* mean in the deviation term (`(x − ewma_v)²` where ewma_v
  already includes x). Standard EWMV sometimes uses the prior mean. Minor bias, low; flag
  for team.
- Plotting helpers (`_colored_panel`, `_build_regime_figure`) are presentation only.

### epdf.py — conditional ePDFs of R_U/R_D by regime (m,n,k)
- `build_epdf`: streams j from j_start; regime read from j−1 EWMA stats; states assigned
  by **expanding-prefix rank** binning (`_state`: equal-count quantile bins via bisect on
  a maintained sorted list, O(n log n)). Genuinely no-lookahead. Good.
- Returns `counts_RU`, `counts_RD` (dict of (m,n,k)→Counter over ℓ), plus *display-only*
  final-history thresholds.
- `_load_1min` here duplicates loading logic (also in plot_volume/app). Minor.
- **[consistency]** the `__main__` demo imports `from plot_volume import _compute_stats`
  and `from ranges import …` — fine under pythonpath, but couples epdf to plot_volume.

### strategy.py — pick_ell_star
- "Largest ℓ* with P(R ≥ ℓ*) ≥ target." Walks ℓ high→low cumulating survival; returns
  first (largest) ℓ meeting target. Logic is correct (survival is monotone in ℓ).
- Edge handling: empty→0, target≤0→max key, target>1→0. Reasonable.
- **[extension]** This is the *entire* strategy surface. Only one knob (fill_rate_target)
  and one policy (static limit at window open, chase at close). Lots of room: chase
  policy, dynamic target, asymmetric U/D logic, cost-aware objective.

### backtest.py — v1 (permissive) and v2 (rolling, no-lookahead)
- **v2 (`run_backtest_rolling`)** is the trustworthy one: same streaming pattern as
  build_epdf, internally consistent (one sorted list used for both state-assignment and
  ePDF accumulation), validated by the day-truncation invariant test. Good.
- **[correctness][v1 subtlety] v1 has a state-assignment mismatch.** `build_epdf`
  populates cells using *expanding-prefix* state assignment, but v1's decision loop
  re-derives states from *final full-history* sorted thresholds (`vol_sorted` etc.). So a
  window can be accumulated into cell A but looked up in cell B. v1 is "permissive by
  design," but this is an *internal inconsistency*, not just lookahead — the headline
  "v1≈v2 within 0.05 ticks" partly launders over it. Worth documenting precisely or
  fixing v1 to use consistent (full-history) assignment for both build and lookup.
- **Fill model**: limit posted ℓ* ticks from open; fills iff R_U≥ℓ* (sell) / R_D≥ℓ*
  (buy), at exactly open±ℓ*·tick; else "chase at close" (close of the window's last bar).
  Reasonable first-order model. Ignores intrabar path/queue priority (fine for this
  assignment).
- **Benchmark = window open.** This makes **TWAP ≡ 0 slippage by construction**
  (`twap_baseline` executes at open). So "strategy vs TWAP" is degenerate; only the VWAP
  comparison is informative. Either switch the benchmark to arrival/VWAP or state clearly
  that TWAP-at-open is the zero line (app.py already says this in a header).
- **[efficiency]** v1 and v2 each recompute `compute_all_ranges` + `compute_ewma_series`
  internally; `run_v1.py` then calls `compute_all_ranges` a 3rd time per side for the
  VWAP t_list. ~6 range passes per market. Compute once, pass arrays in.
- **[consistency]** the fill/slippage block is duplicated verbatim between v1 and v2 (~25
  lines). Factor into a `_simulate_fill(side, ell_star, …)` helper.

### baselines.py — TWAP / VWAP
- `twap_baseline`: returns open prices, zero slippage (see benchmark note).
- `vwap_baseline`: per-window typical-price (H+L+C)/3 volume-weighted. **[efficiency]**
  boolean-mask `.loc` per t → O(n) per window, O(n²) overall on long series. Use the same
  searchsorted/groupby trick as `compute_all_ranges`, or compute VWAP inside the single
  range pass.

### pipeline.py — bridge loader → indexed-by-time frame
- `load_market_indexed`: wraps `load_market`, sets time index. Note it constructs
  `MarketSpec(tick_size=0.0)` — tick is *resolved later* by callers, so the spec's tick
  field is vestigial here. Fine but slightly confusing.

### plot_volume.py — stats + volume figure + standalone tick/day helpers
- **[consistency][correctness] `_compute_stats` defines "active day" as
  `traded_mins ≥ 0.90 × max_traded`** (relative), while `loader.drop_low_liquidity_days`
  uses `≥ 0.90 × 480` (absolute). **Two different liquidity filters in one codebase.**
  The relative one auto-adapts to the contract's session length and is the better
  definition; the loader's absolute 480 is likely wrong for 23h futures. Pick one.
- **[consistency]** tick inference is **duplicated** between `_compute_stats` (l.24-30)
  and `get_tick` (l.68-75) — identical min-price-diff + magnitude-round logic. DRY.
- `plot_volume()` (l.89) calls `_build_figure(csv_path)` with the wrong signature
  (`_build_figure` now takes `(df, ticker, daily, max_traded)`). **[correctness] dead/
  broken function** — would crash if called. Remove or fix.

### app.py — localhost HTML explorer (the parameter-tuning UI deliverable)
- Stdlib `http.server`, renders 5 panels (volume, range hists, regime, ePDF table,
  backtest) with sliders for τ, half-life, M/N/K. No external UI dep. Nice and light.
- **[consistency] UI uses `_load_1min` (single CSV), not the roll-aware loader.** So the
  UI analyzes one contract while `run_v1.py` analyzes the rolled multi-contract series.
  Two different data foundations for "the same" analysis. Decide whether the UI should
  roll too.
- **[consistency]** UI shows only v1 backtest (not v2) + VWAP. Given v1's caveats above,
  the UI is showing the less-trustworthy number.
- No `fill_rate_target` slider despite it being the strategy's main knob.

## Test coverage assessment
- **Strong:** range identity (handcrafted + random), EWMA no-lookahead (tail-shuffle +
  prefix-truncation), v2 backtest day-truncation invariant. These are the right
  invariants and are real.
- **Gaps:**
  - EWMA test mirrors the implementation, not the spec → can't catch the eta[1] skip.
  - No test for `pick_active_contract` roll logic (the crossover behavior).
  - No test pinning the two liquidity definitions (they'd surface the 480-vs-max
    conflict).
  - No test for v1 vs v2 agreement (the headline result is unguarded by CI).
  - No test that `pick_ell_star` survival actually achieves ~target fill rate on a known
    distribution.
  - `strategy`/`baselines`/`ticks` have light or no direct edge-case tests for some paths.

## Priority ranking (my read)
1. **[correctness]** eta[1] skip + the test that hides it. Decide: fix to spec, or
   confirm-and-document with team. Either way, add a *spec-derived* EWMA oracle test.
2. **[consistency/correctness]** the two liquidity definitions (480 absolute vs relative
   max). Unify; the absolute 480 is likely wrong for these futures.
3. **[correctness]** v1 state-assignment mismatch (build vs lookup). Document precisely or
   make consistent.
4. **[architecture]** resolve the package-vs-flat seam. Blocks clean packaging + team PR.
5. **[consistency]** UI data foundation (single CSV vs roll), UI shows v1 not v2, no
   fill-rate slider.
6. **[efficiency]** redundant range passes; O(n²) VWAP. Matters once we sweep params.
7. **[extension]** strategy surface (chase policy, fill-rate sweep, cost-aware objective),
   more markets, v1/v2 agreement test.
8. **[cleanup]** dead `compute_ranges` twin, broken `plot_volume()`, duplicate tick
   inference, duplicate `_load_1min`.

## Stream A findings (2026-05-28) — EWMA/EWMV, ready to paste for the team

**The deviation is a one-index shift, not just a dropped sample.** Algorithm 1
updates step `j` with the *prior* interval `η_{j-1}` ("to avoid forward looking",
per the spec's own note). In 0-based numpy that is `eta[j-2]`. `regime.ewma_ewmv`
instead reads `eta[j-1]` in its `j = 3 … n` loop. Two consequences: (1) `eta[1]`
(the 2nd observation) is never folded in — it is the unique never-read element;
(2) every stored output incorporates the *current* interval's observation, so the
series leads the spec by one step. The one-line fix would be `x = eta[j-2]` (and,
symmetrically, the spec then never folds in the final `eta[n-1]`). Per the team
decision we are **not** changing the loop; the gap is now pinned by
`tests/test_ewma_spec.py` — a spec-derived oracle marked `xfail(strict=True)`, so
it flips to a hard failure the day the loop is corrected — plus an explicit
"`eta[1]` is ignored" regression test. The pre-existing `tests/test_ewma.py` oracle
mirrors the implementation (same skip) and so cannot catch this; that false
confidence is now annotated in its docstring.

**EWMV deviation term (informational — matches spec).** The variance update uses
the *current* mean, `(η_{j-1} − ewma_j)²` with `ewma_j` already updated by the new
point. This is faithful to Algorithm 1 (line 20) but differs from EWMV variants
that use the prior mean; it imparts a small downward bias. No action needed unless
the team prefers the prior-mean convention.
