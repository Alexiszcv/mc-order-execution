# Agent Prompt — mc-order-execution

## Guiding principle

**Write the simplest code that correctly implements the spec.** No abstractions beyond what is needed, no defensive boilerplate, no over-engineering. Prefer flat functions over class hierarchies where a class adds no state. Short files. No comments unless a line would be genuinely confusing without one. If two approaches are equally correct, always pick the simpler one.

---

## Context

Build a **Streamlit application** implementing a volatility/volume-based order management system for futures markets, as specified in the `mc-order-execution/TermProject2_OrderExecution.pdf` file. Data lives in `mc-order-execution/data/`, organized by market (one subdirectory per market, multiple contract CSV files per market).

---

## Data formats

### Market data — OHLC CSV (6 columns, no header)

```
datetime_str, open, high, low, close, volume
# example:
2024.11.28.05:03:00, 134.65, 134.79, 134.60, 134.72, 1823
```

`datetime_str` format: `YYYY.MM.DD.HH:MM:SS`. All files in `data/` follow this format.

### Signal data — external signal CSV (5 columns, no header)

The backtester accepts an optional signal file provided by the user. Its format is:

```
excel_serial_date, hour, minute, price, signal
# example:
45293, 14, 30, 2078.7, 3
```

- `excel_serial_date`: integer, days since 1899-12-30. Convert to datetime via `datetime(1899,12,30) + timedelta(days=d)`.
- `hour`, `minute`: integers defining the bar timestamp (bars are on a 5-minute grid, i.e. `minute` ∈ {0, 5, 10, …, 55}).
- `price`: float, reference price at that bar (single value, not OHLC).
- `signal`: signed integer. Interpretation: positive = BUY pressure, negative = SELL pressure, 0 = flat/no signal. The magnitude can vary by market. The backtester maps this to a directional signal: `signal > 0 → BUY`, `signal < 0 → SELL`, `signal == 0 → skip`.

The signal file is matched to the OHLC data by timestamp. Only bars where a non-zero signal exists trigger an order attempt. If no signal file is provided, the backtester falls back to a simple alternating BUY/SELL mock signal for testing purposes.

**Signal loader**: implement a `SignalLoader` class in `src/signal_loader.py` that:
- Reads the 5-column CSV.
- Reconstructs a datetime index from columns 0, 1, 2.
- Returns a pandas Series indexed by datetime with integer signal values.
- Validates that the signal timestamps align (within tolerance) with the OHLC index after resampling to τ minutes.

---

## Step 1 — Project structure

```
mc-order-execution/
├── README.md
├── app.py                  # Streamlit entry point
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_loader.py       # CSV ingestion, contract stitching, resampling
    ├── signal_loader.py     # External signal file ingestion
    ├── regime.py            # EWMA/EWMV, binning, state assignment
    ├── epdf.py              # Empirical PDF construction and query
    ├── order_manager.py     # Optimal limit level selection
    └── backtester.py        # Simulation and performance metrics
```

---

## Step 2 — `data_loader.py`

Implement a `FuturesLoader` class with the following behavior:

**Contract discovery**
- Given a market directory, glob all `.csv` files.
- All files have 6 columns — no format detection needed.
- Parse datetime column, assign columns `[datetime, open, high, low, close, volume]`.

**Contract stitching**
- For a given market, multiple contracts exist (e.g., RXM25, RXZ25). Roll from one contract to the next when the active contract's volume is systematically overtaken by the next — use a rolling 5-day average volume comparison.
- Alternatively, accept an explicit `roll_dates` dict as parameter.

**Day-level filtering** (per the spec)
- Resample to 1-minute grid for the session hours of the market.
- Discard any calendar day where fewer than 90% of expected 1-minute bars are present.
- Parameter: `min_bar_coverage=0.90`.

**Resampling to τ-minute bars**
- Given a holding period `tau` (in minutes), aggregate 1-min bars: open=first open, high=max high, low=min low, close=last close, volume=sum.
- Return a clean DataFrame indexed by datetime.

**Tick size table** — hardcode a dict:
```python
TICK_SIZES = {
    "ES": 0.25, "NQ": 0.25, "RX": 0.01, "VG": 0.5,
    "BP": 0.0001, "GC": 0.10, "SI": 0.005, "HG": 0.0005,
    "CL": 0.01, "EC": 0.00005, "S": 0.25, "TY": 0.015625,
}
```
Accept `tick_size` as an override parameter so the module is generic for any market.

---

## Step 3 — `regime.py`

Implement `RegimeClassifier` with the following:

**EWMA/EWMV** — implement Algorithm 1 from the spec exactly:
- Parameters: `half_life` (in bars), which determines `lambda = 2^(-1/half_life)`.
- Inputs `eta` can be: range R, rangeUp R_U, rangeDn R_D, volume, or any numeric series.
- Always use `eta[j-1]` at step j to avoid lookahead. Implement as an explicit loop (plain numpy, no pandas ewm — the spec defines a custom recursion).
- Outputs: `ewma` series, `ewmv` series (both same length as input, with NaN for first two steps).

**State assignment (binning)**
- Given an EWMA series and `n_states` (integer), assign each value to a state 1..n_states using quantile-based edges computed on an expanding window (no lookahead).
- Three series to classify: volume state (M states), volatility state (N states), price-change direction state (K states).
- `delta_x[j] = close[j] - close[j-1]`, binned into K states (e.g., K=3: down, flat, up).
- Parameters: `M`, `N`, `K`, all user-configurable.
- Output: a DataFrame with columns `[vol_state, sigma_state, dx_state]`, aligned to the input bars.

---

## Step 4 — `epdf.py`

Implement `EPDFEstimator`:

**Input**: τ-bar DataFrame with OHLC + volume + tick size + state assignments.

**Compute spread counts**:
- For each bar j, compute:
  - `R[j] = round((high - low) / tick_size)`
  - `R_U[j] = round((high - open) / tick_size)`
  - `R_D[j] = round((open - low) / tick_size)`

**Build conditional frequency tables** — online, no lookahead:
- At each bar j (starting from `j_start`, user-defined minimum sample size before estimation begins), increment frequency count for:
  - `counts_R[state_m, state_n, state_k, ell]` using states from bar j-1
  - same for `counts_RU`, `counts_RD`
- Store as nested dicts or sparse arrays (ranges can be up to ~50 spreads for volatile contracts).

**Query**:
- `get_pdf(vol_state, sigma_state, dx_state, quantity='R_U')` → returns dict `{ell: probability}` normalized to sum to 1.
- `get_fill_prob(vol_state, sigma_state, dx_state, ell, quantity='R_U')` → `P(R_U >= ell | state)` = survival function = probability that a limit order placed ell ticks above open gets filled.

---

## Step 5 — `order_manager.py`

Implement `OrderManager`:

**Inputs**: signal direction (`BUY` or `SELL`), current bar's open price, current state (vol, sigma, dx), EPDFEstimator instance, tick size.

**Optimal level selection**:
- For a BUY signal: place limit at `open - ell* × tick`.
  - `ell*` maximizes `ell × P(R_D >= ell | state)`.
  - Or: find largest ell such that `P(R_D >= ell | state) >= fill_rate_threshold`.
- For a SELL signal: symmetric, use R_U, place limit at `open + ell* × tick`.
- Parameters: `fill_rate_threshold` (default 0.5), `max_ell` (cap on ticks to consider).

**Output**: `limit_price`, `ell_star`, `fill_probability`.

---

## Step 6 — `backtester.py`

Implement `Backtester`:

**Inputs**: τ-bar DataFrame, OrderManager, EPDFEstimator, signal Series (from SignalLoader or mock fallback).

**Simulation** (walk-forward, fully out-of-sample):
- For each bar j from `j_start` to end:
  1. Check signal at bar j: if 0 or absent, skip.
  2. Get current state from bar j-1.
  3. Query OrderManager → limit_price.
  4. Determine fill: filled if `low[j] <= limit_price` (BUY) or `high[j] >= limit_price` (SELL).
  5. If filled: record slippage saved = `ell* × tick`, PnL = `close[j] - limit_price` (BUY) or `limit_price - close[j]` (SELL).
  6. If not filled: record as missed.

**Metrics**:
- Fill rate (% of non-zero signal bars where order executed)
- Average slippage saved vs market order (on filled orders)
- PnL series, cumulative PnL
- Sharpe ratio (annualized, assuming τ-minute bars)

**Output**: results dict + per-bar trade log DataFrame.

---

## Step 7 — `app.py` (Streamlit UI)

**Sidebar — parameters**:
- `market`: dropdown (populated from `data/` subdirectories)
- `signal_file`: optional file uploader (5-column CSV as described above)
- `tau`: selectbox [5, 10, 15, 30, 60] minutes
- `half_life`: slider [5, 200] bars
- `M` (vol states): slider [2, 5]
- `N` (sigma states): slider [2, 5]
- `K` (dx states): slider [2, 5], default 3
- `j_start`: number input (minimum bars before estimation starts, default 500)
- `fill_rate_threshold`: slider [0.1, 0.9], default 0.5
- `tick_size_override`: optional number input (leave blank to use table)
- Run button

**Main panel — tabs**:

*Tab 1 — Data overview*
- Line chart: stitched close price series
- Bar chart: daily volume, with discarded days highlighted
- Table: contract roll dates detected

*Tab 2 — Regime analysis*
- Three time-series plots: vol_state, sigma_state, dx_state over time
- Histogram: distribution of each state

*Tab 3 — ePDF viewer*
- State selectors: (vol_state, sigma_state, dx_state) → three dropdowns
- Bar chart: P(R_U = ell | state) and P(R_D = ell | state)
- Line chart: fill probability P(R_U >= ell | state) as function of ell

*Tab 4 — Backtest results*
- Summary metrics table: fill rate, avg slippage saved, Sharpe
- Line chart: cumulative PnL
- Histogram: slippage distribution (filled orders only)
- Trade log: scrollable DataFrame

---

## Step 8 — README.md

Write a README with:
- Project description (2–3 sentences)
- Installation: `pip install -r requirements.txt` then `streamlit run app.py`
- Data layout expected (OHLC CSVs in `data/<market>/`)
- Signal file format specification (5-column CSV, columns: excel_date, hour, minute, price, signal)
- Parameter descriptions (tau, M, N, K, half_life, fill_rate_threshold)
- Brief methodology: EWMA regime classification → conditional ePDF → optimal limit placement → walk-forward backtest

---

## Implementation constraints

- **No lookahead anywhere**: state at bar j always computed from data up to bar j-1.
- **No pandas ewm** for EWMA/EWMV — implement Algorithm 1 explicitly.
- All classes accept `tau`, `tick_size`, `M`, `N`, `K`, `half_life` as constructor parameters.
- No external data fetching — only local CSVs.
- Dependencies: `streamlit`, `pandas`, `numpy`, `plotly`. No sklearn required.
- Python 3.10+.

---

## Validation checks to run

1. `R[j] == R_U[j] + R_D[j]` holds for all bars.
2. EWMV is always non-negative.
3. State assignments are integers in [1, M] / [1, N] / [1, K] with no NaN after j_start.
4. ePDF probabilities sum to 1.0 for any queried state.
5. Signal timestamps align with OHLC index after resampling (warn if >5% mismatch).
6. Backtester fill condition consistent: BUY filled iff `low[j] <= limit_price`.