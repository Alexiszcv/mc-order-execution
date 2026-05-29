# Stream G — Agentic value: execution metrics + in/out-of-sample tuning harness

- **Branch:** `stream/g-agentic`  ·  **Worktree:** `C:/Users/jltch/mc-wt/g`
- **Owns (edits):** new package `src/order_mgmt/agent/` (loader for `AIAgent_*.csv`, metrics,
  harness), new `scripts/run_agent_eval.py`, new `tests/test_agent_*.py`, new
  `notebooks/agent_value.ipynb`, new figures `reports/figures/agent_*.png`, new
  `notes/agent-value.md`.
- **Do NOT touch:** `backtest.py`, `baselines.py`, `strategy.py`, `epdf.py`, `regime.py`,
  `ranges.py`, `loader.py`, `app.py`, `pipeline.py` — **call them read-only**. If you need a
  change in one, stop and flag it as a cross-stream coordination point.
- **Consumes (must not break):** `run_backtest_rolling(...)` (C), `pick_ell_star` + any new
  chase policies (D), `load_market_indexed` / `resolve_tick` (B/pipeline).

## Goal
Make the order-execution module **demonstrably add value to the AI agent's trading**, and
give the teacher a **Kaggle-style harness** to test parameter sets in-sample (IS) vs
out-of-sample (OOS) — ideally with one command / one notebook cell.

## Why (context)
Each market has an `AIAgent_*.csv` — a **5-minute decision series** the "AI agent" trades
(schema: `excel_serial_day, hour, minute, price, vol(=0)`; NOT OHLC — see CLAUDE.md). The
agent decides *what/when* to trade; our module decides *how to fill* each parent order via
the volatility/volume-regime ePDFs + slicer. "Agent value" = how much our execution
improves the agent's realised P&L vs. naïve execution (TWAP/VWAP/market-on-decision). The
open question the user raised: can we push this value higher, and can the teacher *test*
candidate parameter values automatically, IS and OOS, like a leaderboard.

## Tasks
1. **AIAgent loader** (`agent/loader.py`): parse `AIAgent_*.csv` → tidy frame with a real
   datetime index (decode the Excel serial day + hour + minute), price, and the implied
   trade direction/series the agent follows. Inspect the 5th column before assuming it's
   volume (it is 0 in samples). Map each AIAgent file to its market + OHLC contract(s) so the
   regime ePDFs (built from the OHLC bars) can condition the execution.
2. **Value metrics** (`agent/metrics.py`): per parent order and aggregated —
   - realised execution price vs. arrival/decision price (implementation shortfall),
   - slippage in ticks (ℓ units) and in P&L currency,
   - fill rate, unfilled-tail cost,
   - **value-add vs. baselines**: Δ vs. TWAP, VWAP, market-on-decision. This is the headline.
   Report mean AND median (the chase tail makes mean ≈ 0 — see D's finding).
3. **IS/OOS harness** (`agent/harness.py` + `scripts/run_agent_eval.py`): split each
   market's timeline into IS and OOS windows (time-ordered, **no leakage**: ePDFs/regimes
   fit on IS only, evaluated on OOS). Accept a **parameter grid** (τ, half-life m, M/N/K,
   fill_rate_target, chase policy) and produce a leaderboard table: per param set, IS metric
   and OOS metric, across ≥2 markets. Fixed seed. Save a CSV + a figure.
4. **Teacher surface**: the harness must be runnable without editing code —
   `python scripts/run_agent_eval.py --market Gold --grid default` and/or an `ipywidgets`
   cell in `notebooks/agent_value.ipynb`. Document the param schema.

## No-look-ahead (critical)
- IS/OOS split is time-ordered; OOS evaluation reads only IS-fitted ePDFs + bars with
  timestamp `< t`. Add a one-line invariant comment at the split and at the eval loop.
- Reuse the existing no-lookahead backtest (`run_backtest_rolling`) rather than re-deriving
  fills, so the invariant is inherited and tested.

## Tests (`tests/test_agent_*.py`)
- Loader: serial-day decode against a hand-checked timestamp; column schema asserted.
- Metrics: tiny hand-computed scenario → known shortfall / value-add.
- Harness: IS-fit never sees OOS rows (assert on indices); leaderboard shape; determinism
  under fixed seed.

## Done criteria
- Runs end-to-end on **≥2 markets** (genericity), fixed seed, figures in `reports/figures/`.
- Leaderboard reports IS vs OOS value-add vs ≥1 baseline; `pytest -q` green; `ruff check .`
  clean.
- `notes/agent-value.md` summarises **options & trade-offs** for the user/teacher to choose
  param sets from — **do not** hardcode a winning parameter set (CLAUDE.md: user's call).
