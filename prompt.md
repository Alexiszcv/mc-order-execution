# Agent Prompt — mc-order-execution (corrected)

## Guiding principle

**Write the simplest code that correctly implements the spec.** No abstractions beyond what is needed, no defensive boilerplate, no over-engineering. Prefer flat functions over class hierarchies where a class adds no state. Short files. No comments unless a line would be genuinely confusing without one. If two approaches are equally correct, always pick the simpler one.

---

## Context

Build a **Streamlit application** implementing a volatility/volume-based order management system for futures markets, as specified in `TermProject2_OrderExecution.pdf`. Data lives in `mc-order-execution/data/`, organized by market (one subdirectory per market, multiple contract CSV files per market).

---

## Data formats

### Market data — OHLC CSV (6 columns, no header)

```
datetime_str, open, high, low, close, volume
# example:
2024.11.28.05:03:00, 134.65, 134.79, 134.60, 134.72, 1823
```

`datetime_str` format: `YYYY.MM.DD.HH:MM:SS`. All files in `data/` follow this format.

### Signal data — optional external signal CSV (5 columns, no header)

The backtester accepts an optional signal file provided by the user. Its format is:

```
excel_serial_date, hour, minute, price, signal
# example:
45293, 14, 30, 2078.7, 3
```

- `excel_serial_date`: integer, days since 1899-12-30. Convert via `datetime(1899,12,30) + timedelta(days=d)`.
- `hour`, `minute`: integers defining the bar timestamp (bars on a τ-minute grid).
- `price`: float, reference price at that bar.
- `signal`: signed integer. `signal > 0 → BUY`, `signal < 0 → SELL`, `signal == 0 → skip`.

The signal file is matched to the OHLC data by timestamp. Only bars where a non-zero signal exists trigger an order attempt. If no signal file is provided, the backtester falls back to a simple alternating BUY/SELL mock signal for testing purposes.

**Signal loader**: implement a `SignalLoader` class in `src/signal_loader.py` that:
- Reads the 5-column CSV.
- Reconstructs a datetime index from columns 0, 1, 2.
- Returns a pandas Series indexed by datetime with integer signal values.
- Warns (do not raise) if more than 5% of signal timestamps fail to match a bar in the OHLC index after resampling to τ minutes.

---

## Step 1 — Project structure

```
mc-order-execution/
├── README.md
├── app.py                  # Streamlit entry point
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── signal_loader.py
    ├── regime.py
    ├── epdf.py
    ├── order_manager.py
    └── backtester.py
```

---

## Step 2 — `data_loader.py`

Implement a `FuturesLoader` class:

**Contract discovery**
- Given a market directory, glob all `.csv` files.
- Parse datetime column, assign columns `[datetime, open, high, low, close, volume]`.

**Contract stitching**
- For a given market, multiple contracts exist (e.g., RXM25, RXZ25).
- Roll from one contract to the next when the next contract's 5-day rolling average volume persistently exceeds the current contract's 5-day rolling average volume. Detect the crossover date automatically from the data — do not accept a `roll_dates` override parameter, as that would bypass the required logic.

**Day-level filtering**
- Resample to 1-minute grid for the session hours of the market.
- Discard any calendar day where fewer than 90% of expected 1-minute bars are present.
- Parameter: `min_bar_coverage=0.90`.
- Also discard leading days with abnormally low total volume (below 10% of the median daily volume across all days) to exclude illiquid contract periods at inception.

**Resampling to τ-minute bars**
- Given holding period `tau` (minutes), aggregate 1-min bars: open=first, high=max, low=min, close=last, volume=sum.
- Return a clean DataFrame indexed by datetime.

**Tick size table**
```python
TICK_SIZES = {
    "ES": 0.25, "NQ": 0.25, "RX": 0.01, "VG": 0.5,
    "BP": 0.0001, "GC": 0.10, "SI": 0.005, "HG": 0.0005,
    "CL": 0.01, "EC": 0.00005, "S": 0.25, "TY": 0.015625,
}
```
Accept `tick_size` as an override parameter for markets not in the table.

---

## Step 3 — `regime.py`

Implement `RegimeClassifier`:

**EWMA/EWMV — implement Algorithm 1 from the spec exactly.**

Parameters: `half_life` in bars → `lambda_ = 2 ** (-1 / half_life)`.

The recursion below must be implemented as an **explicit Python loop** (no `pandas.ewm`, no `numpy.convolve`):

```
j=1:  sumW=0, sumWX=0, ewma=0, sumWSS=0, ewmv=0
j=2:  sumW=1, sumWX=eta[0], ewma=sumWX/sumW,
      sumWSS=(eta[0]-ewma)**2, ewmv=sqrt(sumWSS/sumW)
j≥3:  sumW  = lambda_*sumW  + 1
      sumWX = lambda_*sumWX + eta[j-1]        # eta[j-1], not eta[j]
      ewma  = sumWX / sumW
      sumWSS= lambda_*sumWSS + (eta[j-1]-ewma)**2
      ewmv  = sqrt(sumWSS / sumW)
```

Key constraint: **always use `eta[j-1]` at step j** — using `eta[j]` would be lookahead.
Output: two numpy arrays `ewma`, `ewmv` of length n, with `nan` for indices 0 and 1.

**State assignment (binning)**

- Input to EWMA/EWMV for volatility: range R of the τ-bar (i.e. `high - low`).
- Input to EWMA/EWMV for volume: bar volume.
- `delta_x[j] = open[j] - open[j-1]` — **use open price**, consistent with the spec's definition of `x_t` as the open price (Section 3.1).
- Assign each EWMA value to a state using **expanding-window quantile edges** (no lookahead):
  - at bar j, compute quantile edges from ewma[2..j-1], then assign ewma[j] to a bin.
  - States are integers in `[1, M]` for volume, `[1, N]` for volatility, `[1, K]` for delta_x.
- Output: DataFrame with columns `[vol_state, sigma_state, dx_state]`, aligned to input bars, with `NaN` before `j_start`.

---

## Step 4 — `epdf.py`

Implement `EPDFEstimator`:

**Spread counts** — for each bar j:
```
R[j]   = round((high[j] - low[j])   / tick_size)
R_U[j] = round((high[j] - open[j])  / tick_size)
R_D[j] = round((open[j] - low[j])   / tick_size)
```
Validation: assert `R[j] == R_U[j] + R_D[j]` for all j (raise if violated by more than 1 tick due to rounding; warn otherwise).

**Build conditional frequency tables** — online, strictly no lookahead:
- At each bar j ≥ `j_start`, read the **state from bar j-1** (vol_state, sigma_state, dx_state).
- Increment:
  - `counts_R[m, n, k, R[j]]   += 1`
  - `counts_RU[m, n, k, R_U[j]] += 1`
  - `counts_RD[m, n, k, R_D[j]] += 1`
- Store as nested dicts: `counts_R[(m, n, k)][ell] = count`.

**Minimum sample guard**: before any query, check that the total count for the requested (m,n,k) triple is ≥ `min_samples` (default 30). If not, return `None` and let the caller handle the fallback (e.g. skip the bar). Do not silently return a distribution estimated on 2 observations.

**Query**:
- `get_pdf(m, n, k, quantity)` → `dict {ell: probability}`, normalized to sum to 1. `quantity` ∈ `{'R', 'R_U', 'R_D'}`.
- `get_fill_prob(m, n, k, ell, quantity)` → `P(quantity >= ell | state)` = survival function = sum of probabilities for all values ≥ ell.

---

## Step 5 — `order_manager.py`

Implement `OrderManager`:

**Inputs**: signal direction (`BUY` or `SELL`), current bar's open price, current state `(m, n, k)`, `EPDFEstimator` instance, tick size.

**Optimal level selection** — the spec leaves the exact criterion open; implement the following two methods and expose `method` as a parameter (`'expected'` or `'threshold'`):

- **`'expected'`** (default): `ell* = argmax_ell [ ell × P(quantity >= ell | state) ]`. This maximizes expected improvement over market order. Iterate ell from 1 to `max_ell`.
- **`'threshold'`**: largest ell such that `P(quantity >= ell | state) >= fill_rate_threshold`.

For a **BUY** signal: use R_D, place limit at `open - ell* × tick`.
For a **SELL** signal: use R_U, place limit at `open + ell* × tick`.

If `get_fill_prob` returns `None` (insufficient data), return `limit_price = open`, `ell_star = 0`, `fill_probability = None` — i.e. fall back to market order.

Parameters: `fill_rate_threshold=0.5`, `max_ell=50`, `method='expected'`.

**Output**: `(limit_price, ell_star, fill_probability)`.

---

## Step 6 — `backtester.py`

Implement `Backtester`:

**Inputs**: τ-bar DataFrame, `OrderManager`, `EPDFEstimator`, signal Series.

**Simulation** — walk-forward, fully out-of-sample:

For each bar j from `j_start` to end:
1. Check signal at bar j: if 0 or absent, skip.
2. Read state from bar j-1: `(vol_state[j-1], sigma_state[j-1], dx_state[j-1])`. If any is NaN, skip.
3. Query `OrderManager` → `(limit_price, ell_star, fill_prob)`.
4. Determine fill:
   - BUY filled iff `low[j] <= limit_price`.
   - SELL filled iff `high[j] >= limit_price`.
5. If filled: `slippage_saved = ell_star × tick`, `PnL = close[j] - limit_price` (BUY) or `limit_price - close[j]` (SELL).
6. If not filled: record as missed trade.

**Metrics**:
- Fill rate: filled / total signal bars.
- Average slippage saved (on filled orders only).
- PnL series and cumulative PnL.
- Sharpe ratio: `mean(PnL) / std(PnL) × sqrt(252 × 6.5 × 60 / tau)`.

**Output**: results dict + per-bar trade log DataFrame with columns `[datetime, direction, open, limit_price, ell_star, filled, PnL, slippage_saved]`.

---

## Step 7 — `app.py` (Streamlit UI)

**Sidebar — parameters**:
- `market`: dropdown (populated from `data/` subdirectories)
- `signal_file`: optional file uploader (5-column CSV)
- `tau`: selectbox [5, 10, 15, 30, 60] minutes
- `half_life`: slider [5, 200] bars
- `M` (vol states): slider [2, 5]
- `N` (sigma states): slider [2, 5]
- `K` (dx states): slider [2, 5], default 3
- `j_start`: number input (minimum bars before estimation, default 500)
- `method`: selectbox [`expected`, `threshold`]
- `fill_rate_threshold`: slider [0.1, 0.9], default 0.5 (shown only when method=`threshold`)
- `max_ell`: number input, default 50
- `tick_size_override`: optional number input (leave blank to use table)
- Run button

**Main panel — tabs**:

*Tab 1 — Data overview*
- Line chart: stitched close price series, with contract boundaries annotated.
- Bar chart: daily volume, with discarded days highlighted in red.
- Table: roll dates detected and days discarded.

*Tab 2 — Regime analysis*
- Three time-series plots: `vol_state`, `sigma_state`, `dx_state` over time.
- Histogram: distribution of each state across all bars.

*Tab 3 — ePDF viewer*
- Three dropdowns: select `(vol_state, sigma_state, dx_state)`.
- Bar chart: `P(R_U = ell | state)` and `P(R_D = ell | state)` side by side.
- Line chart: survival functions `P(R_U >= ell)` and `P(R_D >= ell)` vs ell.
- Display sample count for the selected state; show warning if below `min_samples`.

*Tab 4 — Backtest results*
- Summary metrics table: fill rate, avg slippage saved, Sharpe.
- Line chart: cumulative PnL.
- Histogram: slippage saved distribution (filled orders only).
- Trade log: scrollable DataFrame.

---

## Step 8 — README.md

Write a README with:
- Project description (2–3 sentences).
- Installation: `pip install -r requirements.txt` then `streamlit run app.py`.
- Data layout: OHLC CSVs in `data/<market>/`, one file per contract.
- Signal file format: 5-column CSV (excel_date, hour, minute, price, signal).
- Parameter descriptions: tau, M, N, K, half_life, j_start, method, fill_rate_threshold, max_ell.
- Methodology: EWMA/EWMV regime classification → conditional ePDF → optimal limit level → walk-forward backtest.

---

## Implementation constraints

- **No lookahead anywhere**: state at bar j always computed from data up to bar j-1; eta[j-1] used at step j in EWMA.
- **No `pandas.ewm`** for EWMA/EWMV — implement Algorithm 1 as an explicit loop.
- `delta_x` computed on **open prices**, not close prices.
- All classes accept `tau`, `tick_size`, `M`, `N`, `K`, `half_life` as constructor parameters.
- No external data fetching — only local CSVs.
- Dependencies: `streamlit`, `pandas`, `numpy`, `plotly`. No sklearn.
- Python 3.10+.

---

## Validation checks (run at startup after loading)

1. `R[j] == R_U[j] + R_D[j]` for all bars (warn if off by >1 due to rounding).
2. `ewmv >= 0` everywhere after j=1.
3. State assignments are integers in `[1, M]` / `[1, N]` / `[1, K]` with no NaN after `j_start`.
4. ePDF probabilities sum to 1.0 (±1e-9) for any queried state with sufficient samples.
5. Signal timestamp alignment: warn if >5% of signal bars have no matching OHLC bar.
6. Fill condition consistent: BUY filled iff `low[j] <= limit_price`, SELL iff `high[j] >= limit_price`.