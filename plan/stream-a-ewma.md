# Stream A — EWMA / regime correctness

- **Branch:** `stream/a-ewma`  ·  **Worktree:** `C:/Users/jltch/mc-wt/a`
- **Owns (edits):** `src/regime.py`, `tests/test_ewma.py`, `tests/test_no_lookahead.py`, new `tests/test_ewma_spec.py`
- **Do NOT touch:** anything outside the files above.

## Goal
Pin down the EWMA/EWMV correctness question **without changing the team's recursion**.
Decision already made with the user: *document the deviation + add a spec-derived test
(marked xfail); do not fix the loop* — `regime.py:ewma_ewmv` is the team's function and
needs their sign-off before behavior changes.

## Why (context)
`ewma_ewmv` initialises at j=2 with `eta[0]`, then the loop runs `j = 3..n` reading
`eta[j-1]` (= `eta[2]` on the first iteration). **`eta[1]`, the 2nd observation, is never
folded in.** Spec Algorithm 1 folds in every observation `η_{j-1}`. The numeric effect is
tiny (one dropped sample, decayed away), but it's a real deviation.

The existing `tests/test_ewma.py::_brute_force` **mirrors the source loop** (it also skips
`eta[1]`), so it can only ever prove "code matches its own docstring," never "code matches
the spec." That false confidence is the real problem to fix.

## Tasks
1. In `tests/test_ewma_spec.py` (new), write a **spec-derived** brute-force oracle that
   folds in *every* observation `eta[0], eta[1], …, eta[n-1]` per Algorithm 1 (read the
   spec PDF `Order Management/TermProject2_OrderExecution.pdf`, Algorithm 1, for the exact
   recurrence and the `j-1` indexing). Assert `ewma_ewmv` matches it.
2. Mark that test `@pytest.mark.xfail(reason="regime.ewma_ewmv skips eta[1]; pending team
   confirmation — see notes/component-review.md", strict=True)` so it documents the gap
   *and* will flip to a real failure (alerting us) the day the loop is fixed.
3. Add a focused test asserting the *current* behavior explicitly: that `eta[1]` does not
   affect any output (perturb `eta[1]` only, assert outputs unchanged). This makes the
   skip a documented, intentional-looking property rather than a silent accident.
4. In `regime.py`, add a one-line comment at the loop head naming the deviation and
   pointing to `notes/component-review.md` and the xfail test. **Do not change the loop.**
5. (Optional, document-only) Note in the docstring that EWMV uses the *current* mean in the
   deviation term `(x - ewma_v)²` (ewma_v already includes x), a minor low-bias vs using
   the prior mean. Flag for team; do not change.
6. Keep `test_no_lookahead.py` green; if you add a stronger no-lookahead assertion, do it
   here.

## Done criteria
- `tests/test_ewma_spec.py` exists: an xfail spec oracle + an explicit "eta[1] is ignored"
  test.
- `regime.py` has the one-line deviation comment; recursion unchanged.
- `pytest -q` green (xfail counts as pass), `ruff check .` clean.
- One-paragraph summary of the deviation + the EWMV-mean note appended under a
  "Stream A findings" heading in `notes/component-review.md`, ready to paste into a team
  message.
