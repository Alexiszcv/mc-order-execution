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

### Interactive dashboard (recommended)

```bash
streamlit run src/streamlit_app.py     # → http://localhost:8501
```

A Plotly dashboard with tabs for data quality, ranges, regimes, ePDFs, the
no-lookahead backtest, a fill-target Pareto sweep, and an **Agent value** tab
(see *Agentic value* below). All parameters (τ, half-life, M/N/K, j_start,
fill-rate target) are live sidebar sliders — the parameter-tuning surface.

### Legacy viewer

```bash
python src/app.py                      # → http://localhost:8000
```

Select a contract, adjust the τ slider, then click **Plot**.

### Agent execution-value evaluation

```bash
python scripts/run_agent_eval.py --market Gold --market Nasdaq
```

Scores the AI agent's fills (`AIAgent_*.csv`) against benchmarks and saves
`reports/figures/agent_*.png`. See *Agentic value* below.

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

## Agentic value (AI agent execution)

Each market ships an `AIAgent_*.csv` — a 5-minute decision series for an "AI
agent" (schema `excel_serial_day, hour, minute, price, signed_position`). The
agent decides *what* and *when* to trade; this module decides *how* to fill each
parent order. **Agent value** = how much regime-conditioned execution improves
the agent's realised price versus a naïve benchmark, in ticks
(`src/order_mgmt/agent/`).

**Parent orders & direction.** The agent's trades are the rows where its *signed
position changes*: `dpos = position.diff()`, `dpos > 0` → buy, `< 0` → sell.
Direction comes from the agent's own action, so it is causal — no lookahead.

**Benchmark = the OHLC window open, not the agent's CSV price.** The agent price
can sit on a different (unrolled) contract than our rolled OHLC series; that
basis would otherwise masquerade as slippage (it produced a spurious Gold tail
until we switched the benchmark). Positive ticks therefore mean the execution
layer *beat* market-on-decision.

**No-lookahead split.** The regime ePDFs/thresholds are fit only on OHLC windows
*before* the agent's first trade (`train_end`); every decision is then evaluated
out-of-sample. This is a single time-ordered hold-out — a full in/out-of-sample
parameter leaderboard is the natural next step.

### The headline result

Posting a passive regime limit `ℓ*` earns a strong **median** improvement but a
**mean near zero** — the same chase-on-unfill tail as the main backtest. The
robust win is fill rate and the median; the mean is governed by the tail.

**The winner — regime-limit + chase-cap.** Keep the limit's upside but stop out
at a fixed `cap` ticks of adverse move (`order_mgmt.agent.slicing.fill_capped`).
The asymmetry — uncapped upside, bounded downside — turns the mean positive while
keeping the median and shrinking the 5th-percentile tail:

| Market | Scheme | Mean (ticks) | Median | p5 (tail) |
|--------|--------|--------------|--------|-----------|
| Gold   | regime limit (uncapped) | +0.01 | +4 | −21 |
| Gold   | regime limit + **cap 4** | **+0.99** | +3 | −4 |
| Nasdaq | regime limit + **cap 4** | **+2.97** | +5 | −4 |

(τ=5, half_life=20, M=N=K=3, j_start=200, fill_rate_target=0.6.) The best cap is
market-dependent — the dashboard's **Agent value** tab sweeps it live.

### Strategy levers (Stream D)

Composable refinements on top of the picker, surfaced as options (the final
parameter choice is the user's):

- **Cost-aware ℓ\*** (`pick_ell_star_cost_aware`) — maximise
  `p(ℓ)·ℓ − (1−p(ℓ))·chase_cost` instead of targeting a fill rate; never picks
  worse than market-on-open.
- **Chase-at-mid** (`chase_price(policy="mid")`) — fill unfilled orders at the
  window mid `(H+L)/2` rather than the close; strictly better mean *and* tail.
- **Early-chase** (`simulate_early_chase`) — bail when price moves a trigger
  distance against the limit instead of waiting for the deadline (the intrabar
  cousin of the chase-cap).

### Genericity, not over-fitting

A regime-conditioning ablation (`scripts/sweep_chase.py`, `notes/strategy-sweep.md`)
found the 27-cell edge over a single *pooled* ePDF is only ~0.02–0.05 ticks of
mean — the chase-cap, not the conditioning, is doing the heavy lifting. The
pipeline is generic across all provided markets (`scripts/run_agent_eval.py`
accepts any `--market`), and a `synthetic` agent generator
(`order_mgmt.agent.synthetic`) provides a zero-shot genericity check.

Full findings: `notes/agent-value.md` (execution value) and
`notes/strategy-sweep.md` (Stream D sweep + ablation).

---

## Project layout

```
src/
  streamlit_app.py    Plotly dashboard (port 8501) — data quality, ranges,
                      regimes, ePDFs, backtest, sweep, Agent value tab
  viz_plotly.py       Plotly figure builders for the dashboard
  app.py              Legacy web viewer (port 8000)
  ranges.py           τ-window range computation (compute_ranges, compute_all_ranges)
  epdf.py             Conditional ePDF builder (build_epdf) + raw CSV loader
  plotting.py         Range histogram figure
  plot_volume.py      Daily-volume figure + 90%-of-max liquidity filter + tick inference
  regime.py           EWMA / EWMV recursion + regime visualisation
  order_mgmt/
    loader.py         Contract-roll-aware market loader (load_market, MarketSpec)
    pipeline.py       Bridges load_market into the indexed-by-time format
    ticks.py          Per-market spec tick-size table + resolver
    strategy.py       pick_ell_star + cost-aware/random pickers, chase_price,
                      simulate_early_chase (Stream D)
    baselines.py      TWAP / VWAP baselines
    backtest.py       run_backtest (v1) + run_backtest_rolling (v2, no-lookahead)
    mc/               Monte Carlo execution layer (Stream F)
    agent/            AI-agent execution value (Stream G): loader, metrics,
                      benchmarks, slicing (fill_capped), dynamic (DP), synthetic

tests/                pytest — math primitives, strategy, ticks, pipeline, loader,
                      MC, agent
scripts/
  run_v1.py           End-to-end backtest demo: Gold + Nasdaq, v1 vs v2 vs VWAP
  run_agent_eval.py   Agent execution-value eval per market
  sweep_*.py          Fill-rate Pareto + chase-policy sweeps (Stream D)
  compare_*.py        Agent benchmark / slicing / dynamic / tail-cap comparisons
reports/figures/      Backtest + agent + sweep figures
```