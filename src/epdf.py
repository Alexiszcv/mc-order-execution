"""Conditional empirical PDFs for R_U and R_D, conditioned on regime state (m, n, k).

Range computation lives in `ranges.py`; histogram plotting lives in `plotting.py`.
"""

from __future__ import annotations

import bisect
from collections import Counter

import numpy as np
import pandas as pd


def build_epdf(t_list, ell_u_list, ell_d_list, ewma_vol, ewma_range, dx_list,
               M: int, N: int, K: int, j_start: int):
    """
    Build empirical PDFs of R_U and R_D conditioned on regime state (m, n, k).

    At step j, the regime is read from the *previous* window (index j-1):
    ewma_vol[j-1], ewma_range[j-1], dx_list[j-1].  Each indicator is classified
    into {1..S} states using quantile thresholds computed on the expanding prefix
    ewma_vol[:j-1] (all values strictly before j-1).

    State assignment uses bisect on a maintained sorted list → O(n log n) total.

    Parameters
    ----------
    t_list       : window start timestamps (length n)
    ell_u_list   : R_U in ticks per window (length n)
    ell_d_list   : R_D in ticks per window (length n)
    ewma_vol     : EWMA of volume aligned on t_list, NaN at head (length n)
    ewma_range   : EWMA of range  aligned on t_list, NaN at head (length n)
    dx_list      : Δx in ticks aligned on t_list, NaN at discontinuities (length n)
    M            : number of volume states  {1..M}
    N            : number of range  states  {1..N}
    K            : number of Δx     states  {1..K}
    j_start      : first index to start accumulating (must be >= 1)

    Returns
    -------
    counts_RU  : dict {(m, n, k): Counter}   keyed by (vol_state, range_state, dx_state)
    counts_RD  : dict {(m, n, k): Counter}
    thresholds : dict {'vol': list, 'range': list, 'dx': list}
                 Final quantile thresholds over the full series (for display only —
                 not used by the binning, which uses an expanding prefix).
    """
    n  = len(t_list)
    ev = np.asarray(ewma_vol,   dtype=float)
    er = np.asarray(ewma_range, dtype=float)
    dx = np.asarray(dx_list,    dtype=float)

    counts_RU: dict = {}
    counts_RD: dict = {}

    def _state(value: float, sorted_arr: list, n_states: int):
        """State in {1..n_states} from the rank of value in sorted_arr."""
        L = len(sorted_arr)
        if L < n_states:
            return None
        rank = bisect.bisect_left(sorted_arr, value)
        return min(rank * n_states // L + 1, n_states)

    # Pre-populate sorted lists with the prefix strictly before j_start - 1
    # so that at j = j_start, sorted lists hold ev[0..j_start-2].
    pre = max(0, j_start - 1)
    sv = sorted(float(v) for v in ev[:pre] if not np.isnan(v))
    sr = sorted(float(v) for v in er[:pre] if not np.isnan(v))
    sd = sorted(float(v) for v in dx[:pre] if not np.isnan(v))

    for j in range(max(j_start, 1), n):
        v_cur = ev[j - 1]
        r_cur = er[j - 1]
        d_cur = dx[j - 1]

        if not (np.isnan(v_cur) or np.isnan(r_cur) or np.isnan(d_cur)):
            m    = _state(float(v_cur), sv, M)
            n_st = _state(float(r_cur), sr, N)
            k    = _state(float(d_cur), sd, K)

            if m is not None and n_st is not None and k is not None:
                key = (m, n_st, k)
                if key not in counts_RU:
                    counts_RU[key] = Counter()
                    counts_RD[key] = Counter()
                counts_RU[key][ell_u_list[j]] += 1
                counts_RD[key][ell_d_list[j]] += 1

        # Insert current values so they're available at the next step.
        if not np.isnan(v_cur):
            bisect.insort(sv, float(v_cur))
        if not np.isnan(r_cur):
            bisect.insort(sr, float(r_cur))
        if not np.isnan(d_cur):
            bisect.insort(sd, float(d_cur))

    # Final quantile thresholds on the full series (display-only).
    def _final_qs(arr: np.ndarray, n_states: int) -> list:
        valid = arr[~np.isnan(arr)]
        if len(valid) < n_states:
            return []
        return np.percentile(
            valid, [100.0 * s / n_states for s in range(1, n_states)]
        ).tolist()

    thresholds = {
        "vol":   _final_qs(ev, M),
        "range": _final_qs(er, N),
        "dx":    _final_qs(dx, K),
    }

    return counts_RU, counts_RD, thresholds


def _load_1min(csv_path: str) -> pd.DataFrame:
    """Load a raw CSV file into a 1-minute bar DataFrame indexed by datetime."""
    return pd.read_csv(
        csv_path, header=None,
        names=["time", "open", "high", "low", "close", "volume"],
        index_col="time",
        parse_dates=["time"],
        date_format="%Y.%m.%d.%H:%M:%S",
    )


if __name__ == "__main__":
    from pathlib import Path

    from plot_volume import _compute_stats
    from ranges import compute_all_ranges

    csv = Path(__file__).parents[1] / "data" / "Gold" / "GCG24.csv"
    if not csv.exists():
        print(f"data file not found: {csv}")
        raise SystemExit(0)

    df_1min = _load_1min(str(csv))
    _, tick, proper_days, n_green, n_total = _compute_stats(df_1min)
    print(f"loaded {csv.stem}: {len(df_1min)} bars, {n_green}/{n_total} active days, tick={tick:g}")

    t_list, ell_r, ell_u, ell_d, _vol, _dx = compute_all_ranges(
        df_1min, tau=5, tick=tick, proper_days_list=proper_days
    )
    print(f"computed {len(ell_r)} τ=5 windows")
    if ell_r:
        mism = sum(1 for r, u, d in zip(ell_r, ell_u, ell_d) if abs(r - (u + d)) > 1)
        print(f"R = R_U + R_D within ±1 tick: {len(ell_r) - mism}/{len(ell_r)}")
