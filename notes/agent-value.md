# AIAgent execution-value pipeline (Stream G, slice: metrics + zero-shot)

How well does the regime-conditioned limit-order execution serve the "AI agent's"
trades? This pipeline answers that automatically, on any market and on synthetic
data. Code lives in `src/order_mgmt/agent/`; the runnable surface is
`scripts/run_agent_eval.py`.

## The data finding that drives everything

`AIAgent_*.csv` has **no header**. Schema (assigned by us, verified against the data):
`day, hour, minute, price, extra`.

- `day` is an **Excel serial day** (base `1899-12-30`): `45293 → 2024-01-02`.
- **`extra` is the agent's running signed position (inventory)** — Gold takes integer
  values −7..+8 across the file. It is *not* volume (CLAUDE.md's guess) and *not* a
  binary flag.
- The **parent orders are the rows where the position changes**: `dpos = position.diff()`;
  `dpos>0` → buy, `dpos<0` → sell, `dpos==0` → no order. Gold: **1,276 trades / 31,057 rows**.
- Direction comes from `Δposition` (the agent's own action, known at decision time) →
  **no lookahead**. Inferring direction from `price[i+1]−price[i]` would be lookahead and
  is deliberately rejected (asserted in `tests/test_agent_loader.py`).

## What we measure — and what we do NOT

There is **no order-book / quote data**, so we cannot and do not model bid-ask spread
cost or market impact. We measure **implementation shortfall**: realized execution
price vs the agent's arrival (decision) price, in ticks, under the project's
regime-conditioned limit-order policy.

- We post a passive limit `ℓ*` ticks on the favorable side of the execution window open.
  From OHLC alone we know whether price traversed that level inside `[t, t+τ)`: if the
  realized half-range reaches `ℓ*` the limit fills `ℓ*` ticks better than arrival;
  otherwise we **chase** at the window close.
- **Sign convention:** positive ticks = the strategy beat the benchmark (sold higher /
  bought lower).
- **CAVEAT (stated in code too):** the fill model is *optimistic* — touching the limit
  level is assumed to fill (no queue position, no partial fills, no spread). These are
  execution-quality numbers from OHLC, **not** a P&L claim.

### Read the median, not the mean

The mean shortfall is dragged toward (or below) zero by the **unfilled-chase tail**:
the minority of orders that don't fill get chased to the window close and can lose tens
of ticks. The **median** is the honest headline. Representative run (defaults τ=5,
m=20, M=N=K=3, fill-rate target 0.6, `j_start=200`):

| market | source    |    n | fill |  mean | median | unfilled-tail |
|--------|-----------|-----:|-----:|------:|-------:|--------------:|
| Gold   | real      |  948 | 75.5%| −0.27 |  +4.00 |        −14.9  |
| Gold   | synthetic |  249 | 67.9%| +0.57 |  +1.00 |         −3.8  |
| Nasdaq | real      | 1568 | 89.2%| −1.27 |  +5.00 |        −51.0  |
| Nasdaq | synthetic |  263 | 61.2%| +2.46 |  +3.00 |        −11.8  |

So: on the typical trade the patient limit captures ~4–5 ticks of improvement, while a
small unfilled tail does the damage. Whether that net trade-off "adds value" depends on
how you weight the tail — which is exactly the parameter choice below.

## No-lookahead

ePDFs + regime quantile thresholds are fit on OHLC windows with start time `< train_end`
(default = the first agent decision; for Gold/Nasdaq the agent trades in 2024 with years
of prior OHLC history). The same frozen thresholds bin both the training outcomes and the
eval decisions (build/lookup consistency). Each decision reads regime only from the latest
OHLC window **closed before** `t`. `# INVARIANT` comments mark the fit boundary and the
per-decision read in `metrics.py`.

## Parameter knobs (no winning set is chosen — that's the user's/teacher's call)

`evaluate_agent_execution(..., tau, half_life, M, N, K, fill_rate_target, j_start, train_end)`:

- **`tau` (holding period, min)** — the execution window. Longer τ → more chance to fill
  the limit (higher fill rate) but more drift risk on the chase. Spec set: {5,10,15,30,60}.
- **`half_life` m** — EWMA/EWMV memory for the regime indicators. Short m reacts fast but
  is noisy; long m is smoother but laggier.
- **`M, N, K`** — number of volume / range / Δprice regime bins. More bins = sharper
  conditioning but thinner per-cell ePDFs (more decisions skipped for empty cells).
- **`fill_rate_target`** — how aggressively `pick_ell_star` posts. Higher target → smaller
  `ℓ*` → fills more often but captures fewer ticks; lower target → larger `ℓ*` → bigger
  improvement when filled, fatter unfilled tail. **This is the main lever on the
  mean-vs-median trade-off above.**
- **`train_end`** — the no-lookahead train/eval boundary; defaults to the first decision.

## Zero-shot / new-data testing

`synth_agent_series(df_ohlcv, *, seed, n_decisions, mode, max_abs_position, start_fraction)`
emits the **same `AgentSeries` shape** as the real loader, so it runs the identical
metrics path — proving the pipeline is generic, not Gold-overfit. Decisions are confined
to the last `1−start_fraction` of the OHLC coverage so the regime fit has prior history.
Two `mode`s: `sample_closes` (arrival = OHLC close at-or-before each decision) and
`random_walk` (seeded Gaussian). Everything is seeded for reproducibility.

```bash
python scripts/run_agent_eval.py --market Gold --market Nasdaq --synthetic --seed 0
```

To evaluate a *new real* AIAgent file, drop it into a market folder as `AIAgent_*.csv`
(schema `day,hour,minute,price,extra`) and point `--market` at that folder.

## Order sizing: agent sets side, our model can set size

The agent supplies side + timing; `metrics.size_weighted_shortfall(fills, rule)` re-weights
the *same* per-decision shortfalls by a model-driven size (all weights ex-ante → no lookahead):

- **`agent`** — weight = `|Δposition|` (baseline: the model sizes nothing, only sets `ℓ*`).
- **`confidence`** — weight = `fill_prob · ℓ*` = expected ticks captured (size ∝ the model's edge).
- **`inverse_vol`** — weight = `1/σ̄` proxy (smaller in volatile regimes; risk-based).

`scripts/sweep_agent_metrics.py` sweeps `fill_rate_target × τ` and emits two figures per
market: `agent_sweep_<m>.png` (general metrics) and `agent_sizing_<m>.png` (the three rules).

**Finding (Gold + Nasdaq): no sizing rule dominates — it is market- and τ-dependent.**
- **agent-size** weighting is usually the *worst* (often net-negative, badly so at τ=30): the
  agent trades biggest exactly when passive execution is hardest (large moves → chase tail).
- **inverse-vol** is the most *consistent* (positive at short/mid τ in both markets, rarely worst)
  — "trade bigger when calm" aligns with passive limits working best in calm regimes.
- **confidence** is *high-variance*: it can be the best (Nasdaq τ=15, +7–9 ticks) or the worst
  (Nasdaq τ=10/30, −3 to −4 ticks), because it concentrates size on a few large-`ℓ*` orders.

The size-weighted numbers are small (≈ ±1–4 ticks) and the grids are jagged → treat
cross-cell differences cautiously; this is direction-of-effect, not a tuned result. **Pick a
rule out-of-sample** (the deferred IS/OOS leaderboard is the right place to rank them).

## Slicing a parent order ("buy 5 in the next 5 min")

The agent emits parent orders like "buy 5 over the next τ min"; `agent/slicing.py` cuts each
one (all **within** τ — the agent re-decides every 5 min, so we must not stretch across
windows). `metrics.evaluate_agent_schemes` runs every scheme on the *same* orders / regime /
`ℓ*`, so the only difference is how the order is cut. `scripts/compare_slicing.py` →
`agent_slicing_compare_<m>.png` + table.

- **single** — one limit for the whole order (current).
- **time_slice (K=3)** — split into K children across sub-intervals; each posts a √(1/K)-scaled limit.
- **blend (f=0.5)** — fill (1−f) at the open immediately (certain), post f as a limit.
- **cutoff (0.5)** — post the limit; if unfilled halfway through, market the remainder then.

**Finding (Gold + Nasdaq, τ=5, target 0.6): it's a median-vs-tail frontier, no free lunch.**

| scheme | Gold median / p5-tail | Nasdaq median / p5-tail |
|---|---|---|
| single | **+4.0** / −50.7 | **+5.0** / −38.6 |
| time_slice K=3 | +0.7 / −37.7 | +0.0 / −55.9 |
| blend f=0.5 | +1.5 / **−25.6** | +2.0 / **−19.0** |
| cutoff 0.5 | +3.0 / −30.6 | +4.0 / −35.0 |

- **single-shot** maximises the median (+4/+5) but has the **fattest tail** (all-or-nothing).
- **blend** roughly **halves the tail** (−50→−26 Gold, −39→−19 Nasdaq) and fills the most, but
  surrenders most of the median.
- **cutoff** is the best **all-rounder**: keeps the median near single-shot (+3/+4) while
  trimming the p5 tail ~40%.
- **time-slice (textbook TWAP-style) is the worst here** — it dilutes the median *and* (Nasdaq)
  worsens the tail; √K-scaled child limits barely improve while chases still bite.
- The **deep tail (p1 ≈ −220 Gold)** is untouched by any within-window scheme: a window-wide
  price spike hits every scheme that still executes inside τ. Only a shorter τ or not trading
  escapes it.

**No rule is "best" — pick by risk appetite:** lowest average price → single/cutoff; smallest
chance of a bad fill → blend; balance → cutoff. Rank out-of-sample before committing.

## Dynamic (sequential) execution: re-decide each minute

`agent/dynamic.py` treats "buy Q in τ min" as optimal stopping: each minute with r left,
post a limit ℓ from the current price; fill → done; else carry to r−1; deadline → market.
Two policies, both fit on training windows (no lookahead) then simulated on the real bars:

- **adaptive** — ℓ_r = largest offset with r-minute fill-prob ≥ target (shrinks as r→1).
- **dp** — backward induction `V(r)=max_ℓ[q₁(ℓ)·ℓ + (1−q₁(ℓ))·(V(r−1)−w)]`, w = per-cell
  expected 1-min adverse excursion (the one explicit modelling knob).

`scripts/compare_dynamic.py` → `agent_dynamic_compare_<m>.png`. Results (τ=5, target 0.6):

| scheme | Gold median / p5 | Nasdaq median / p5 |
|---|---|---|
| single | **+4.0** / −50.7 | **+5.0** / −38.6 |
| blend | +1.5 / −25.6 | +2.0 / −19.0 |
| cutoff | +3.0 / −30.6 | +4.0 / −35.0 |
| adaptive | +3.0 / −34.6 | +4.0 / −32.0 |
| dp | +1.0 / −31.6 | +1.0 / **−9.0** |

**Finding: the policies optimise different objectives, so "best" depends on the objective.**
- **DP wins on mean and tail** (the risk metrics): best (least-negative) mean on both markets,
  and on Nasdaq it cuts the worst-5% from −38.6 → **−9.0** ticks. By getting conservative near
  the deadline it almost never gets caught in a bad chase. It pays for this with the **lowest
  median** (+1) — it is literally optimising expected cost, not the typical fill.
- **adaptive (rule) captures ~80% of the benefit cheaply**: keeps the median near single-shot
  (+3/+4) while trimming the tail — the pragmatic middle.
- **single-shot keeps the best median but the worst tail** (greedy/all-or-nothing).
- Gold's **deep p1 (≈ −223)** is untouched by every policy — a window-wide spike hits anything
  executing inside τ. Nasdaq's tail is timing-driven, so DP recovers it (p1 −98 → −58).

So: minimise expected/worst-case cost → **DP**; best typical price → **single**; robust balance
→ **adaptive**. The DP's objective (and its wait-penalty w) is a knob — change it to target the
median instead of the mean if that's the goal.

## DATA-QUALITY FIX: benchmark vs the OHLC open (basis-immune)

Early metrics benchmarked execution against the **agent's CSV price**, which sits on a
*different contract* at rolls (e.g. Gold 2024-03-26/27, 05-29/30: agent ≈ OHLC open − 220
ticks). That booked the **contract-roll basis** as fake slippage and inflated the tail
(Gold p5 went −21 → −50, p1 → −223 purely from basis; Nasdaq was unaffected — no basis).
Diagnosed by `scripts/diagnose_tail.py` (windows are clean: 0 cross-midnight, 0 gaps, all
5-bar). **Fix:** every execution metric now benchmarks vs the **OHLC window open** — the
price on the instrument we actually fill, and the standard implementation-shortfall arrival.
All tables below are post-fix.

## Benchmarks: is the method actually good? (`agent/benchmarks.py`)

Same orders, no-/low-skill baselines, basis-immune. `scripts/compare_benchmarks.py`.

| strategy | Gold mean/median/p5 | Nasdaq mean/median/p5 |
|---|---|---|
| market (all-in) | 0 / 0 / 0 | 0 / 0 / 0 |
| random offset | −0.15 / +2.0 / −16.0 | −1.36 / +3.0 / −27.0 |
| global L* (no regime) | +0.08 / +3.0 / −15.6 | −1.16 / +3.0 / −24.6 |
| **regime L\* (ours)** | +0.01 / **+4.0** / −21.0 | −1.33 / **+5.0** / −39.6 |
| DP (complex) | −0.30 / +1.0 / −7.0 | −0.62 / +1.0 / **−8.0** |

(market ≡ the open benchmark, so it is 0/0/0 — everything is improvement-over-market.)

**Findings:** (1) the skill ordering on the **median** is monotone and correct — market 0 <
random +2 < global +3 < **regime +4/+5**; the ablation `regime > global` passes, so regime
conditioning earns ~1 tick over no-regime and ~2 over guessing. (2) But naive regime limits
give the median back through the chase **tail** (mean ≈ 0 Gold, negative Nasdaq). Only the DP
controls the tail among the clever strategies.

## The winner: regime limit + chase-cap (`evaluate_tail_strategies`)

A tight **chase-cap** (post the regime ℓ*, but market out at `cap` ticks of adverse move)
is asymmetric — keeps the limit upside, truncates the tail. `scripts/compare_tail_reduction.py`:

| strategy | Gold mean/median/p5 | Nasdaq mean/median/p5 |
|---|---|---|
| market | 0 / 0 / 0 | 0 / 0 / 0 |
| regime (no cap) | +0.01 / +4 / −21.0 | −1.33 / +5 / −39.6 |
| **cap = 4** | **+0.99 / +3 / −4.0** | **+2.97 / +5 / −4.0** |
| cap = 8 | +0.40 / +4 / −8.0 | +2.29 / +5 / −8.0 |

**This is the headline result:** the chase-capped regime limit **beats market on the mean AND
the median while holding a tail near market's** (cap 4: Gold +0.99 mean / +3 median / −4 tail;
Nasdaq +2.97 / +5 / −4). Tighter cap ⇒ better mean & tail (down to ~4 ticks here). It is the
one strategy that dominates naive execution on all three axes — and it's just the regime ePDF
(for the offset) plus a stop (for the tail).

## Deferred (next slice)

The Kaggle-style **IS/OOS leaderboard** over a parameter grid is not built here. The
single time-ordered train/eval split (`train_end`) is the seam it will plug into.

## Open modeling decisions (surfaced, not decided)

- **Order size:** primary metrics are per-order ticks; `captured_improvement_notional`
  weights by `|Δposition|` and is labeled captured improvement, **not** an impact cost
  (we have no order book to price impact).
- **Opening position:** the first AIAgent row has no prior `dpos`, so the opening
  inventory is treated as pre-existing, not a trade.
