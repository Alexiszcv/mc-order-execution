# Plan folder — parallel work streams

Each `stream-*.md` is a **self-contained brief** for one Claude Code session working in
its own git worktree. A fresh agent should be able to open the file cold and execute it
without reading the others. Detailed code findings these plans are built on live in
`../notes/component-review.md`.

## How to run a stream (worktree recipe)

```bash
# 1. From the main repo on research-tchimby, commit this plan/ folder FIRST so every
#    worktree inherits it.
git add plan/ notes/component-review.md && git commit -m "docs(plan): parallel work streams"

# 2. Create a worktree + branch per stream you want to run NOW. Put them OUTSIDE OneDrive
#    (OneDrive sync locks .git files). Example target: C:\Users\jltch\mc-wt\<stream>
git worktree add C:/Users/jltch/mc-wt/a  stream/a-ewma
git worktree add C:/Users/jltch/mc-wt/c  stream/c-backtest
git worktree add C:/Users/jltch/mc-wt/d  stream/d-strategy
git worktree add C:/Users/jltch/mc-wt/e  stream/e-ui

# 3. Open a separate Claude Code session in each folder. Point it at the matching
#    plan/stream-*.md. They run fully in parallel.

# 4. When a stream is green (pytest + ruff), merge it back:
git switch research-tchimby
git merge --no-ff stream/a-ewma
git worktree remove C:/Users/jltch/mc-wt/a    # cleanup
```

**Per-worktree environment:** reuse the root `.venv` (deps are identical across
branches). From inside a worktree, run `pytest -q` and `ruff check .` — pytest detects the
worktree's own `pyproject.toml` as rootdir and puts *that worktree's* `src` on the path, so
you test the worktree's code, not the root's. If you hit an import surprise, create a
throwaway venv in the worktree (`uv venv && uv pip install -e ".[dev]"`).

**`data/` is committed**, so every worktree has the CSVs — no symlink needed.

## Stream map & dependency graph

| Stream | Branch | Owns (edits) | Consumes (read-only) |
|---|---|---|---|
| A — EWMA correctness | `stream/a-ewma` | `regime.py`, `test_ewma.py`, `test_no_lookahead.py`, `test_ewma_spec.py`(new) | — |
| B — data / liquidity | `stream/b-data` | `loader.py`, `plot_volume.py`, `ticks.py`, `test_loader.py` | — |
| C — backtest | `stream/c-backtest` | `backtest.py`, `baselines.py`, `test_backtest.py` | strategy, ranges, epdf, regime |
| D — strategy research | `stream/d-strategy` | `strategy.py`, `scripts/sweep_*.py`(new), `test_strategy.py` | backtest, baselines |
| E — UI alignment | `stream/e-ui` | `app.py`, `plotting.py` | pipeline, plot_volume, backtest |

Streams **A, B, C, D, E touch disjoint files** and may run concurrently. Read-only
consumers depend on the **shared contracts** below — owners must not break them silently.

## Shared contracts (do NOT change signatures without announcing in the merge PR)

These are the cross-stream API surfaces. If your stream must change one, it becomes a
coordination point — flag it, and the consuming streams rebase after.

- `plot_volume._compute_stats(df) -> (daily, tick, proper_days, n_green, n_total)` —
  owned by **B**, consumed by C, E. If B changes the return shape, C & E must rebase.
- `order_mgmt.backtest.run_backtest(...)` and `run_backtest_rolling(...)` keyword
  signatures — owned by **C**, consumed by D, E, `scripts/run_v1.py`. **C must only ADD
  optional kwargs**, never rename/remove existing ones.
- `order_mgmt.strategy.pick_ell_star(epdf, fill_rate_target) -> int` — owned by **D**.
  New policies must be NEW functions; do not change this signature (C/backtest calls it).
- `order_mgmt.pipeline.load_market_indexed(market_dir, *, min_fraction)` — used by E.

## Conventions (all streams)
- **No look-ahead, ever** (CLAUDE.md). Any new recursive/streaming code reads only `j-1`
  and earlier; add a one-line invariant comment if not obvious.
- **Refactor commits ≠ behavior commits.** Keep structure-only and output-changing changes
  in separate commits so merges are reviewable.
- **Done = `pytest -q` green + `ruff check .` clean + runs on ≥2 markets** where
  applicable. Save any new figure to `reports/figures/` with a fixed seed.
- Append any new cross-cutting finding to `notes/component-review.md` so the other streams
  see it on their next rebase.

## Deferred / not-now
- **Stream 0 — package/flat unification.** Move flat `src/*.py` into `order_mgmt/`, rewrite
  all imports. This rewrites every file → cannot run alongside A–E. Do it as a single
  dedicated session AFTER the others merge, or skip to stay team-PR-compatible. Not
  scheduled yet.
