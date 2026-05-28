# Next session pickup (saved 2026-05-27)

## Where we are

Branch `research-tchimby`, **10 commits ahead of `origin/research-tchimby`**, nothing pushed yet.
Tests **32 passing**. Working tree clean except `Order Management/` (untracked
leftover from the old folder layout — safe to delete).

```
cd7be40  feat(phase-b-c-v2): integration, polish, no-lookahead backtest
5023494  feat(phase-a): tests, strategy, backtest, baselines, demo, UI ePDF panel
1af5374  fix(epdf): convert per-day index to ns before int64 arithmetic
2476edd  Merge team/main into research-tchimby as Phase A foundation
8e71477  chore(claude): clean up settings.json /doctor warnings
```

## What got done this session

- **Phase A** — tests, strategy (`pick_ell_star`), v1 backtest, TWAP/VWAP baselines,
  two-market demo, UI ePDF panel.
- **Phase B** — roll-aware pipeline, tick-size table, hygiene (dead code, requirements
  sync), module split (`ranges.py`, `plotting.py`, `epdf.py`).
- **Phase C** — UI backtest panel, README results section, French → English in
  `regime.py`.
- **v2 stretch** — `run_backtest_rolling`: strict no-lookahead streaming variant.
  Tested via day-truncation invariant on real Gold data.

Also: a one-line **pandas-3.0 fix** in `compute_all_ranges` (`asi8` returned μs not
ns under the new default datetime unit, breaking everything; team should probably
PR this back).

## Headline result

v1 vs v2 agree within 0.05 ticks on the mean → lookahead bias is small at this
config. The strategy shows:

- **Median** +2 to +7 ticks vs TWAP (typical fill saves several ticks)
- **Mean** near zero (chase-on-unfill tail eats most of the gain)
- **Fill rate** 66–72% (close to the 0.6 target)

Memory: `memory/v1_v2_lookahead_finding.md`.

## Possible next moves (pick one)

1. **Push to `origin/research-tchimby`** — preserves your work remotely. `git push
   origin research-tchimby`.
2. **Open a PR against team `main`** — `gh pr create --base main --head
   research-tchimby` against `Alexiszcv/mc-order-execution`. Talk to the team about
   the pandas-3.0 fix and the `ewma_ewmv` skipped-eta[1] quirk before merging.
3. **Tighten the strategy** — sweep `fill_rate_target ∈ {0.4, 0.5, 0.6, 0.7, 0.8}`,
   plot the Pareto curve of mean-slippage vs fill-rate. Should show whether the
   sweet spot is somewhere other than 0.6.
4. **Smarter chase policy** — current v1/v2 chase at window close. Try
   "chase at mid" or "chase as soon as price moves >X ticks against the limit"
   and see if the negative tail shrinks.
5. **Address the `ewma_ewmv` quirk** — the function skips `eta[1]` in
   initialisation. Document and either fix or confirm with the team that the
   docstring's behaviour is what they want.
6. **Run on more markets** — Bunds (RX), EuroStoxx (VG), GBP (BP) are present
   in `data/`. Add them to `MARKETS` in `scripts/run_v1.py`.

## Open questions

- Is the `Order Management/` directory still needed? It's the pre-merge folder
  layout with the assignment PDF. The team has `TermProject2_OrderExecution.pdf`
  at the root now. Either delete or copy any local research-only files out
  first.
- The team's `ewma_ewmv` skips `eta[1]` in init — is this intended or a bug?
  Their docstring says it is; the spec wants every observation folded in. Worth
  a Slack message before any "fix".
- Do you want to push your work to your personal remote (`personal/main`) as
  well as the team repo's research branch? You haven't touched `personal/main`
  in this session — its tip is at `3e133c6 ci(claude): use
  CLAUDE_CODE_OAUTH_TOKEN instead of API key`.

## Useful commands to resume with

```bash
# Re-run the demo and re-generate figures
python scripts/run_v1.py

# Re-run all tests
pytest -q

# See what's ahead of remote
git log --oneline origin/research-tchimby..HEAD

# See what's in the team's main vs your branch
git log --oneline origin/main..HEAD
```
