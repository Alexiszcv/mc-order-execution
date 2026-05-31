# Monte Carlo — Volatility-Volume-based Order Management

Columbia IEOR4703 (Hirsa) Term Project 2: slice parent orders into child orders that get filled with high probability while minimizing slippage, by conditioning execution on volatility/volume regimes mined from 1-min OHLC futures data. The deliverable is a generic, back-tested module that works across all provided futures markets, not a one-off script. Full assignment spec lives in `Order Management/TermProject2_OrderExecution.pdf` — that PDF is the source of truth for definitions and requirements.

## Stack

- **Language/runtime:** Python 3.11+
- **Core libs:** numpy, pandas, scipy, matplotlib (statsmodels if needed for regime tests)
- **Notebooks:** Jupyter for exploration; final logic must live in `.py` modules, not notebooks
- **Package manager:** uv (preferred) or pip + `requirements.txt`
- **No framework** — this is a research/quant codebase, not a service

## Where things live

- `Order Management/TermProject2_OrderExecution.pdf` — assignment spec, **read this first**
- `Order Management/data/<Market>/` — 1-min OHLC+volume CSVs per futures contract (e.g. `GCM24.csv`)
  - Format: `YYYY.MM.DD.HH:MM:SS, open, high, low, close, volume` (no header)
  - `AIAgent_*.csv` files use a different schema (`day, hour, minute, price, ?`) — inspect before joining
- `src/` (TBD) — reusable modules: data loading, range/rangeUp/rangeDn, EWMA/EWMV regime binning, ePDFs, order slicer, backtester
- `notebooks/` (TBD) — exploratory analysis, plots, one notebook per market or per question
- `tests/` (TBD) — pytest unit tests for math primitives (range formulas, EWMA recursion, no-lookahead invariants)
- `reports/` (TBD) — final write-up + figures
- `README.md` (TBD) — **final-submission deliverable**, ships with the Jupyter notebooks. Write near the end; grows with the project.
- UI (TBD, **final-submission deliverable**) — `ipywidgets` cells embedded in the final notebook is the lightest path; `app.py` with Streamlit is fine if a standalone is preferred

## Commands

```bash
# Install
uv sync           # or: pip install -r requirements.txt

# Run a notebook
jupyter lab

# Test (run before claiming a task is done)
pytest -q

# Lint / format
ruff check . && ruff format .
```

## Domain definitions (from the spec — use these exactly)

- **Holding period** τ ∈ {5, 10, 15, 30, 60} min; user-specified, code must accept any of these.
- **Tick size** ε: per-market constant (ES = 0.25; look up the rest in the contract spec — do **not** hardcode without a source).
- **Range** `R_{t,τ} = H − L = ℓ·ε`. **RangeUp** `R^U = H − O = ℓ₁·ε`. **RangeDn** `R^D = O − L = ℓ₂·ε`. Express results in units of ε (counts of spreads), not raw price.
- **EWMA / EWMV**: half-life parameterization `λ = 2^(−1/m)` — see Algorithm 1 in the spec. At step `j`, use `η_{j−1}` (previous interval), never `η_j`, to avoid forward-looking.
- **State conditioning**: bin running volume / volatility / Δprice into M·N·K regimes, then compute ePDFs of range/rangeUp/rangeDn per regime.

## Conventions

- **No look-ahead, ever.** Every recursive update at step `j` reads only `j−1` and earlier. Every backtest signal at time `t` reads only data with timestamp `< t`. If this isn't obvious at a glance, add a one-line comment naming the invariant.
- **Generic across markets.** Loader/feature/strategy code takes a market spec (tick size, symbol, file pattern) as input. No `if symbol == "ES"` branches in core logic.
- **Express prices in spreads (ℓ = ΔPrice/ε)** for distributions and rule thresholds; convert back to price only at the boundary.
- **Reproducibility:** seed every RNG explicitly; commit notebooks with outputs cleared and the seed at the top.
- **Naming:** snake_case for functions/vars, PascalCase for classes, modules kebab-or-snake. Match what's already in the file you're editing.
- **One module per concept** (`ranges.py`, `ewma.py`, `epdf.py`, `slicer.py`, `backtest.py`) — keep files small and composable so they can be unit-tested in isolation.

## Gotchas

- **Data quality:** the first weeks/months of a contract have thin trading and should be dropped. Drop any day with < 90% of expected trading minutes. Eyeball volume per Figure 1 of the spec before trusting a contract.
- **Contract rolls:** each market has multiple expiry CSVs (e.g. `GCG24`, `GCJ24`, `GCM24` for Gold). Use each contract only during its high-liquidity window before expiry; do not naively concatenate. See spec Figure 1 (ESH20 → ESM20 around 2020-03-17) for the pattern.
- **Non-overlapping ranges:** when comparing `R_{t,τ}` to a prior range `R_{t−δ,τ}`, ensure `[t−δ, t−δ+τ]` does not overlap `[t, t+τ]`. The spec calls this out explicitly.
- **`AIAgent_*.csv` ≠ OHLC CSV:** different schema. Don't auto-detect; pass the loader a file type.
- **Tick size varies per market** — ES is 0.25, but Bunds, Gold, JPY, EuroStoxx all differ. Hardcoding 0.25 will silently corrupt distributions for every non-ES market.
- **Plots in the spec are MATLAB** — replicate the *content*, not the styling. Use matplotlib with sensible defaults.

## Don't

- Don't introduce look-ahead bias, even transiently for "easier debugging." If you need it to validate, gate it behind an explicit `allow_lookahead=True` arg and never set it true in backtest paths.
- Don't commit raw CSV data into git if a repo is later initialized — the `data/` tree is large and not ours to redistribute. Use `.gitignore`.
- Don't edit `Order Management/TermProject2_OrderExecution.pdf` — it's the assignment handout.
- Don't add a heavy framework (Backtrader, Zipline, vectorbt) without asking — the assignment expects a self-contained module.
- Don't write 200-line notebook cells. If logic is reusable, lift it into `src/` and import it.

## Working principles

How we collaborate on this codebase. These are deliberately short — extras live in `~/OneDrive/Bureau/claude_knowledge/guides/` and `/cookbooks/`.

- **Plan before implementing** anything that touches >2 modules or changes a public function signature. Use `/plan`, get approval, then `/implement`. Don't skip on "small" changes that have hidden ripples (e.g. anything affecting the no-lookahead invariant).
- **Vertical slices.** Wire one full path through (load → range/ewma → epdf → backtest) on ONE market before broadening to all 10. Don't write the whole feature library, then the whole backtester.
- **Test math primitives before trusting them.** Unit-test the identities (`R = R^U + R^D`), the EWMA recursion against a brute-force reference, and the no-lookahead invariant. These are cheap to write and catch the bugs that silently corrupt distributions.
- **Bug fixes: reproduce → failing test → root cause → fix cause, not symptom.** A "results look weird" report needs a deterministic repro (fixed seed, fixed slice of data) before any code change. Use `/debug` if the cause isn't obvious. Symptoms hidden by null checks come back worse.
- **Refactors are separate commits from behavior changes.** A commit either changes *structure* or changes *output*, never both. Mixed commits are unreviewable.
- **Notebooks are scratch; modules are truth.** If a notebook cell exceeds ~50 lines or you copy logic between notebooks, lift it into `src/`. The final write-up should import from `src/`, not redefine.
- **Self-review the diff before claiming done.** Read it cold. Especially watch for accidental look-ahead, hardcoded tick sizes, and overlapping windows.
- **End-of-session ritual.** Tests green, WIP committed or stashed, next step noted in one line.

### When to delegate

- **Survey questions across files** ("where do we compute volume regimes?") → `code-explorer` subagent so raw output doesn't bloat the main thread.
- **Independent review of a finished piece** → `code-reviewer` subagent. Catches what you've gone blind to.
- **Designing a non-trivial change** → `code-architect` subagent before writing code.
- **Deep bug investigation** → `debugger` subagent; returns diagnosis + proposed fix, doesn't apply it.
- **Trivial lookups, single greps, reading one file** → just do it inline; subagents aren't for one tool call.

### How to brief me

- State **goal + constraints + done criteria** — not "look at X" but "X should produce Y, must not break Z, done when test W passes."
- **Don't delegate understanding.** If you have a hypothesis about where the bug is or which approach you want, say so. "Use your judgment" usually means "I haven't decided" — better to ask for a recommendation explicitly.
- **Pick a scope verb:** Explain / Plan / Sketch / Implement. Mixing them wastes turns.
- **Say what you've ruled out.** Saves me re-exploring.

## Definition of done

A task is done when:
- [ ] `pytest -q` passes (math primitives have tests: range identity `R = R^U + R^D`, EWMA recursion vs. brute-force reference, no-lookahead invariant)
- [ ] `ruff check` is clean
- [ ] The code runs end-to-end on **at least two different markets** (proves the generic-module requirement)
- [ ] Any new ePDF / regime / strategy is reproduced with a fixed seed and the figure is saved to `reports/figures/`
- [ ] Backtest results (when applicable) report slippage and fill-rate vs. a naïve TWAP/VWAP baseline

## Final submission

Two artifacts beyond the modules/notebooks, both due with the final submission:

- **`README.md`** — bundled with the Jupyter notebooks. Should describe what the project does, how to install (`pip install -e ".[dev]"`), how to run the notebooks, where the UI lives, and a results summary. Write this near the end — it evolves with the project, not the scaffold.
- **Parameter-tuning UI** — a simple interactive surface to vary τ, ε, half-life `m`, M/N/K regime counts, and roll thresholds, then re-run the analysis without editing code. Default choice: `ipywidgets` cells in the final notebook (no separate app needed; lives in one submission file). Streamlit/Gradio is fine if you want it standalone.

These ship in the **final** vertical slice — slice 10, after the backtester works. Don't start them earlier; the parameter set isn't stable until then.

## Out of scope for Claude

- Don't pick a final strategy / parameter set on the user's behalf — surface options with tradeoffs and let them choose.
- Don't fetch external market data or hit broker APIs — work only with the CSVs in `Order Management/data/`.
- Don't `git init` or commit anything unless explicitly asked (this directory is not yet a repo).
- Don't bump dependency major versions without asking.
