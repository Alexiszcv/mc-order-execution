# mc-order-execution

Volatility/volume-based order management system for futures markets.
Empirical PDFs of intrawindow price ranges, conditioned on market regime, are used
to optimally place limit orders and minimize slippage.

---

## Installation

```bash
# Editable install with dev dependencies (pytest, ruff, jupyter)
pip install -e ".[dev]"
# Minimal install (runtime only):
pip install -r requirements.txt
```

## Usage

### Interactive viewer

```bash
python src/app.py
```

Open `http://localhost:8000` in your browser. Select a contract from the
dropdown, adjust the τ slider, then click **Plot**.

### End-to-end backtest demo

```bash
python scripts/run_v1.py
```

Loads each market's full contract history (rolled at daily-volume crossover),
runs the regime-conditioned strategy on both sides (buy / sell), compares to
TWAP and VWAP baselines, prints a slippage + fill-rate summary, and saves
histograms to `reports/figures/`.

### Tests

```bash
pytest -q
```

Math primitives (range identity `R = R_U + R_D`, EWMA recursion vs.
brute-force reference, no-lookahead invariant) plus strategy and tick-table
unit tests.

---

## Interface

The viewer displays two plots:

**Volume chart** — 1-minute traded volume over the full contract history.
Days are shaded green (kept) or red (discarded) based on the data quality
filter described below.

**Range distributions** — histograms of R, R_U, R_D in ticks for the
selected τ, computed only on the kept days.

---

**Tick size** is inferred automatically as the minimum non-zero difference between distinct prices observed in the contract history. This is a heuristic, not an exact rule — it works well in practice but can be overridden via the `tick_size_override` parameter if the inferred value is incorrect.

---

## Methodology

### Day filtering

A day is kept if the number of 1-minute bars with at least one trade is at least
90% of the maximum observed across all days for that contract. Days below this
threshold are discarded (shown in red) and excluded from all subsequent
computations. The reference maximum is the single busiest day in the contract
history.

### Valid windows

For a given holding period τ (in minutes), the session is sliced into
consecutive non-overlapping windows `[t, t+τ)`, `[t+τ, t+2τ)`, … starting from
the first bar of the day. A window is valid if and only if:

- it contains exactly τ 1-minute bars (no missing bar inside the window), and
- it does not cross midnight (the window ends on the same calendar day it starts).

Incomplete windows — those at the end of a session or around intraday gaps —
are discarded.

### Range quantities

For each valid window the following quantities are computed in price space and
in tick space (ℓ = round(value / tick)):

| Symbol | Formula | Meaning |
|--------|---------|---------|
| R | max(high) − min(low) | Full bar range |
| R_U | max(high) − open₀ | Upward half-range |
| R_D | open₀ − min(low) | Downward half-range |

where open₀ is the open of the first 1-minute bar in the window.
By construction R = R_U + R_D (up to rounding artefacts of ±1 tick).

### Volatility proxy

Two options are available for estimating the current volatility regime:

- **EWMA of range** — the exponentially weighted moving average of R over past
  windows. More natural: directly measures the average recent range.
- **EWMV of range** — the exponentially weighted moving variance of R. Captures
  the stability of volatility rather than its level (volatility of volatility).

Both are computed via the recursive Algorithm 1 from the project spec, using
`η_{j-1}` at step j to avoid any lookahead. The half-life m (in bars) is a
user parameter.

### Δx discontinuities

Since only a subset of days is retained, consecutive kept days are not necessarily contiguous in calendar time. The first bar of each kept day is therefore assigned `Δx = NaN` to avoid contaminating the direction signal with overnight or multi-day price jumps. These bars are excluded from the binning threshold computation and do not increment the conditional frequency tables.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| τ | Holding period in minutes | 5 |
| half_life | EWMA/EWMV half-life in bars | 20 |
| M | Number of volume regime states | 3 |
| N | Number of volatility regime states | 3 |
| K | Number of price-direction states | 3 |
| j_start | Minimum bars before ePDF estimation begins | 200 |
| fill_rate_target | Minimum fill probability when picking ℓ\* | 0.6 |

---

## Strategy and backtest

At each τ-window decision point:

1. Classify the prior window into a regime cell `(m, n, k)` from EWMA-volume,
   EWMA-range, and Δx quantile states.
2. Look up the cell's empirical PDF of R_U (for a sell) or R_D (for a buy).
3. Pick **ℓ\*** = largest tick distance such that `P(R ≥ ℓ\*) ≥ fill_rate_target`
   (`order_mgmt.strategy.pick_ell_star`).
4. Place a limit order at `open ± ℓ\* · tick`. If the realized R_U/R_D meets ℓ\*,
   the order fills at the limit price; otherwise it chases at the window's close.

Slippage is reported in ticks vs. a TWAP baseline (market-execute at open).
The VWAP baseline is also computed for context.

### Results (`scripts/run_v1.py`)

Settings: τ=5, half_life=20, M=N=K=3, j_start=200, fill_rate_target=0.6.
Full contract history per market (roll-aware loader).
Two backtest variants reported:

- **v1** — uses ePDFs built from the *full* history (lookahead permitted; the
  "what's the maximum edge under perfect knowledge of marginal distributions"
  upper bound).
- **v2** — strict no-lookahead. At each decision *j*, ePDFs and quantile
  thresholds are built incrementally from data strictly before *j*.
  (`run_backtest_rolling`)

| Market | Contracts rolled | Side | Variant | n | Fill rate | Avg (ticks) | Median (ticks) |
|--------|------------------|------|---------|----|-----------|-------------|----------------|
| Gold   | GCG24 / GCJ24 / GCM24 / GCQ24 | buy  | v1   | 35 437 | 66.9% | +0.06 | +2 |
| Gold   | GCG24 / GCJ24 / GCM24 / GCQ24 | buy  | v2   | 35 410 | 66.1% | +0.04 | +3 |
| Gold   | GCG24 / GCJ24 / GCM24 / GCQ24 | sell | v1   | 35 437 | 66.9% | +0.19 | +3 |
| Gold   | GCG24 / GCJ24 / GCM24 / GCQ24 | sell | v2   | 35 410 | 66.1% | +0.18 | +3 |
| Nasdaq | NQH20 / NQM20 / NQU20         | buy  | v1   | 39 391 | 68.6% | −0.17 | +6 |
| Nasdaq | NQH20 / NQM20 / NQU20         | buy  | v2   | 39 364 | 71.9% | −0.12 | +6 |
| Nasdaq | NQH20 / NQM20 / NQU20         | sell | v1   | 39 391 | 68.9% | +0.17 | +6 |
| Nasdaq | NQH20 / NQM20 / NQU20         | sell | v2   | 39 364 | 71.3% | +0.12 | +7 |

VWAP baseline averages are ±0.04 ticks on Gold and ±0.03 on Nasdaq —
essentially flat against TWAP=open at the τ=5 horizon.

**Interpretation.** v1 and v2 are within 0.05 ticks on the mean; the lookahead
bias is small in this configuration. Both variants show:

- **Strongly positive median** — the typical fill saves +2 to +7 ticks vs. TWAP
- **Mean near zero** — dragged down by the chase-on-unfilled tail
- **Fill rate 66–72%** — close to the 0.6 target

The edge is real but modest. Tightening `fill_rate_target`, refining regime
granularity, or adopting a smarter chase policy are the obvious levers.

### Known simplifications

- **VWAP execution assumption.** The baseline assumes you can transact at the
  bar-typical-price (high + low + close) / 3 weighted by bar volume.
  Optimistic; real VWAP execution has implementation shortfall.
- **Chase price = window close.** Unfilled orders are charged at the close
  of the τ-window. Real execution might allow earlier intervention or pay
  half-spread, both of which would tighten the slippage tails.
- **Tick size from heuristic by default.** For known markets the spec value
  from `order_mgmt.ticks.TICK_TABLE` overrides the inferred minimum-price-
  difference heuristic (`plot_volume.get_tick`). Unknown markets fall back to
  the heuristic.

---

## Project layout

```
src/
  app.py              Web viewer (port 8000) — volume, range histograms,
                      regime indicators, conditional ePDFs, backtest panel
  ranges.py           τ-window range computation (compute_ranges, compute_all_ranges)
  epdf.py             Conditional ePDF builder (build_epdf) + raw CSV loader
  plotting.py         Range histogram figure
  plot_volume.py      Daily-volume figure + 90%-of-max liquidity filter + tick inference
  regime.py           EWMA / EWMV recursion + regime visualisation
  order_mgmt/
    loader.py         Contract-roll-aware market loader (load_market, MarketSpec)
    pipeline.py       Bridges load_market into the indexed-by-time format
    ticks.py          Per-market spec tick-size table + resolver
    strategy.py       pick_ell_star (limit-distance from ePDF + fill target)
    baselines.py      TWAP / VWAP baselines
    backtest.py       run_backtest (v1) + run_backtest_rolling (v2, no-lookahead)

tests/                pytest — math primitives, strategy, ticks, pipeline, loader
scripts/run_v1.py     End-to-end demo: Gold + Nasdaq, v1 vs v2 vs VWAP
reports/figures/      Backtest figures (slippage histograms per market)
```