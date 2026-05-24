# mc-order-execution

Volatility/volume-based order management system for futures markets.
Empirical PDFs of intrawindow price ranges, conditioned on market regime, are used
to optimally place limit orders and minimize slippage.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/app.py
```

Open `http://localhost:8000` in your browser. Select a contract from the
dropdown, adjust the τ slider, then click **Plot**.

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
| j_start | Minimum bars before ePDF estimation begins | 500 |
| fill_rate_threshold | Minimum fill probability to place a limit order | 0.5 |