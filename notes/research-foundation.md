# Research Foundation — Volatility-Volume-based Order Management

Literature survey conducted 2026-05-15. Use this as the starting point for the related-work
section, and to inform method-design decisions for slices 4–9.

## 1. Formal problem name

**Optimal Trade Execution** — a subfield of market microstructure / algorithmic trading.
Splits into two flavors:

1. **Schedule-level execution** — given a parent order of size X and horizon T, choose the
   *time profile* of child orders to minimize total cost (slippage + risk).
   Almgren-Chriss / Bertsimas-Lo formulation.
2. **Order-placement execution** — given that you're sending a child order *now*, choose its
   *type* (market vs. limit) and *aggressiveness* (how many ticks inside/outside the spread).
   Cont-Stoikov / Avellaneda-Stoikov / LOB-stochastic flavor.

**This project (Hirsa T2) lives in camp 2.** It builds regime-conditional empirical
fill-probability distributions from OHLC data and uses them as a rule — a non-parametric
cousin of Cont-Stoikov, with state-dependence borrowed from regime-switching volatility
literature.

## 2. Three schools of solutions

### A. Optimization-based (closed-form / DP)

- **Bertsimas & Lo (1998), "Optimal Control of Execution Costs"** — Bellman/DP; first formal
  treatment. `https://www.mit.edu/~dbertsim/papers/Finance/Optimal%20control%20of%20execution%20costs.pdf`
- **Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions"** — THE canonical
  paper. Linear permanent + temporary impact, mean-variance objective, closed-form optimal
  trajectory. Every desk's IS algorithm descends from this.
  `https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf`
- **Cartea, Jaimungal, Penalva (2015),** *Algorithmic and High-Frequency Trading* — modern
  stochastic-control textbook. **Best single source for this problem.**
  `https://api.pageplace.de/preview/DT0400.9781316455579_A25606943/preview-9781316455579_A25606943.pdf`

### B. LOB stochastic models (closest to this project's approach)

- **Cont, Stoikov, Talreja (2010), "A Stochastic Model for Order Book Dynamics"** —
  birth-death Poisson model; fill probabilities computed semi-analytically via Laplace
  transforms. Computes exactly the kind of quantity Hirsa's Figure 2 estimates empirically.
  `http://www.columbia.edu/~ww2040/orderbook.pdf`
- **Avellaneda & Stoikov (2008)** — market-making framework with reservation prices.
- **Sun et al. (2024), "Fill Probabilities in a LOB with State-Dependent Stochastic Order
  Flows"** — generalization to state-dependent intensities. **Spiritually identical to what
  we're doing**, just from the LOB side rather than OHLC.
  `https://arxiv.org/abs/2403.02572`

### C. Learning-based

- **Nevmyvaka, Feng, Kearns (2006), "RL for Optimized Trade Execution"** — first RL for
  execution (Q-learning on NASDAQ LOB).
  `https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf`
- **Hendricks & Wilcox (2014)** — Almgren-Chriss + Q-learning hybrid.
- **Ning et al. (2018)** — DQN; **Lin & Beling (2020)** — PPO with sparse rewards.
- **Frey et al. (2024, arxiv 2411.06389)** — multi-agent simulator, current frontier.

## 3. Industry baselines

| Algo | What it does | When it wins | When it loses |
|---|---|---|---|
| **TWAP** | Equal time-slicing | Liquid, low-vol, "set and forget" | Sharp intraday vol moves |
| **VWAP** | Slices proportional to expected intraday volume profile | Highly liquid, predictable U-shape volume | Atypical volume days |
| **POV** | Constant fraction of *live* volume | Volatile / illiquid markets | Forces participation in toxic flow |
| **IS (Almgren-Chriss)** | Minimizes cost+risk vs. arrival price | Impact params well-calibrated | Mis-specified impact → expensive |

Practitioner consensus: VWAP remains the reporting benchmark because every desk uses it.
IS is the *right* answer when impact is calibrated. POV is the robust fallback. RL
outperforms in narrow regimes but is brittle to regime shifts.

## 4. Where this project sits — what's novel

This is a **regime-conditional, non-parametric, rule-based, limit-order placement** strategy:

- **Non-parametric** — no Almgren-Chriss impact model required; distributions estimated directly.
- **Regime-conditional** — `M·N·K` states from (volume, volatility, Δprice) via EWMA/EWMV.
  Captures non-stationarity that pooled distributions miss.
- **Limit-order-aware** — Figure 2 of the spec is literally `P(fill | spread, regime)`, the
  same quantity Cont-Stoikov computes parametrically.
- **Bayesian framing** — spec hints at using `P(market goes up by ℓ spreads | state)` as a
  likelihood. Underspecified in the handout — **this is the biggest design degree of freedom we own**.

Published evidence in this family:

- State-conditioned fill probability dominates pooled (Sun et al. 2024; also visible in
  Hirsa's Figure 2 segment-3 vs. segments 1–2).
- Adaptive participation beats fixed schedules when volume forecasts are noisy.
- Hybrid IS + learning (Hendricks-Wilcox style) currently best published execution cost.

## 5. Improvement opportunities (ranked by leverage)

1. **Add full baselines.** Backtest against TWAP + VWAP + Almgren-Chriss IS, not just naïve
   TWAP. Spec mandates only *a* baseline — adding IS gives a publishable comparison.
2. **KDE the ePDFs** instead of raw histograms. Smoother, less binning bias, especially on
   thin tails (segment-3 territory).
3. **Cross-validate `M, N, K`** instead of eyeballing. Walk-forward time-series CV gives a
   principled regime-count.
4. **Tighten the Bayesian update.** Spec waves at "Bayesian inference"; specify a Beta prior
   on fill probability (Bernoulli fills → conjugate), update on every observation.
5. **Numerical no-lookahead test.** Randomize timestamp order and verify outputs change.
   Cheap insurance against silent leaks.
6. **(Stretch)** Train a small tabular Q-learner on the same regime representation; compare
   to the rule. Modern thing to do; exposes whether the ePDF rule leaves money on the table.

## 6. Recommended reading order

1. **Cartea-Jaimungal-Penalva**, chapters on optimal execution + LOB — best single source.
2. **Almgren-Chriss 2000** — read once; we won't use it directly but we'll benchmark against it.
3. **Cont-Stoikov-Talreja 2010** — maps directly to the fill-probability side of our work.
4. **Sun et al. 2024** (state-dependent fill probabilities) — most aligned with our approach.
5. **Nevmyvaka-Kearns 2006** — read after the rule-based version works, to motivate the RL stretch.

## 7. Next-session pickup

- **Repo:** `https://github.com/tchimby/monte-carlo-order-management` (private)
- **Routine:** monte-carlo-weekly-planner (`trig_01Q8ThBiY4RXQ8gawfg5oZMr`), Monday 09:00 ET
- **Venv:** `.venv/` (Python 3.14, all deps installed via `pip install -e ".[dev]"`)
- **Last slice landed:** Slice 1 — data loader (`src/order_mgmt/loader.py`, 7 tests green)
- **Next slice:** Slice 2 — `ranges.py` (R, R^U, R^D in spreads; identity `R = R^U + R^D`)
- **Open methodological decisions:**
  - KDE vs. binned histogram for ePDFs (slice 3)
  - Bayesian update prior on fill probability (slice 7)
  - Whether to add Almgren-Chriss IS as a baseline (slice 8)
- **Stretch items to flag at slice 9:** README + ipywidgets UI (final-submission deliverables, see `CLAUDE.md → Final submission`).
