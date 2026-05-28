"""Range computation per τ-window: R, R_U, R_D, plus volume and Δx.

Split out of `epdf.py` so range computation and ePDF construction are separate concerns
(CLAUDE.md: one concept per module). Backward-compatible re-exports remain in `epdf.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ranges(
    df_1min: pd.DataFrame,
    tau: int,
    tick: float,
    proper_days_list,
    quantity: str,
    t_start=None,
    t_end=None,
):
    """
    For each validated day, iterate explicitly over consecutive τ-minute windows
    [t, t+τ[, [t+τ, t+2τ[, … starting from the first bar of the day.

    A window is valid when it contains exactly τ 1-minute bars and does not
    cross midnight.  Optional t_start / t_end clip which windows are kept.

    Parameters
    ----------
    df_1min          : 1-minute bar DataFrame indexed by datetime
    tau              : window length in minutes
    tick             : tick size of the market
    proper_days_list : iterable of dates (date / Timestamp / str) to process
    quantity         : 'R', 'R_U', or 'R_D'
    t_start          : ignore windows whose start < t_start (default: −∞)
    t_end            : ignore windows whose start ≥ t_end   (default: +∞)

    Returns
    -------
    t_list     : list of window-start Timestamps
    value_list : list of price-space values (float)
    ell_list   : list of tick-space values  (int)
    """
    if quantity not in ("R", "R_U", "R_D"):
        raise ValueError(f"quantity must be 'R', 'R_U', or 'R_D', got {quantity!r}")

    dt_tau = pd.Timedelta(minutes=tau)
    dt_last = pd.Timedelta(minutes=tau - 1)

    t_list, value_list, ell_list, vol_list = [], [], [], []

    for day in proper_days_list:
        day_ts = pd.Timestamp(day).normalize()
        day_bars = df_1min[df_1min.index.normalize() == day_ts]
        if day_bars.empty:
            continue

        t = day_bars.index[0]

        while t <= day_bars.index[-1]:
            if (t + dt_last).date() != t.date():
                break

            if t_start is not None and t < pd.Timestamp(t_start):
                t += dt_tau
                continue
            if t_end is not None and t >= pd.Timestamp(t_end):
                t += dt_tau
                continue

            t_next = t + dt_tau
            mask = (day_bars.index >= t) & (day_bars.index < t_next)
            window = day_bars.loc[mask]

            if len(window) == tau:
                hi = window["high"].max()
                lo = window["low"].min()
                first_open = window["open"].iloc[0]

                if quantity == "R":
                    value = hi - lo
                elif quantity == "R_U":
                    value = hi - first_open
                else:  # R_D
                    value = first_open - lo

                t_list.append(t)
                value_list.append(float(value))
                ell_list.append(int(round(value / tick)))
                vol_list.append(float(window["volume"].sum()))

            t += dt_tau

    return t_list, value_list, ell_list, vol_list


def compute_all_ranges(
    df_1min: pd.DataFrame,
    tau: int,
    tick: float,
    proper_days_list,
    t_start=None,
    t_end=None,
):
    """
    Single-pass computation of R, R_U, R_D for all validated days.

    Optimisations vs compute_ranges:
      - pre-groups by date once (O(n_total) once, not × n_days)
      - works on raw int64 nanosecond arrays → np.searchsorted in O(log n)
      - no pandas slicing overhead in the inner loop

    Returns
    -------
    t_list, ell_r, ell_u, ell_d, vol_list, dx_list  — six parallel lists
    dx_list : open[i+1] - open[i] in ticks; NaN when windows i and i+1 are not adjacent
    """
    NS_PER_DAY = 86_400 * 10**9
    dt_tau_ns = int(pd.Timedelta(minutes=tau).value)
    dt_last_ns = int(pd.Timedelta(minutes=tau - 1).value)
    ts_start_ns = int(pd.Timestamp(t_start).value) if t_start is not None else None
    ts_end_ns = int(pd.Timestamp(t_end).value) if t_end is not None else None

    # Force-convert each group's index to ns so the int64 arithmetic below
    # is unit-correct on pandas 3.0+ (where the default datetime unit is us, not ns).
    proper_set = {pd.Timestamp(d).normalize() for d in proper_days_list}
    day_data = {}
    for day_ts, grp in df_1min.groupby(df_1min.index.normalize()):
        if day_ts in proper_set:
            day_data[day_ts] = (
                grp.index.as_unit("ns").asi8,
                grp["high"].to_numpy(),
                grp["low"].to_numpy(),
                grp["open"].to_numpy(),
                grp["volume"].to_numpy(),
            )

    t_list, ell_r, ell_u, ell_d, vol_list, op_first_list = [], [], [], [], [], []

    for day in proper_days_list:
        day_ts = pd.Timestamp(day).normalize()
        if day_ts not in day_data:
            continue
        idx_ns, hi_arr, lo_arr, op_arr, vol_arr = day_data[day_ts]

        t_ns = int(idx_ns[0])
        end_ns = int(idx_ns[-1])

        while t_ns <= end_ns:
            if (t_ns + dt_last_ns) // NS_PER_DAY != t_ns // NS_PER_DAY:
                break
            if ts_start_ns is not None and t_ns < ts_start_ns:
                t_ns += dt_tau_ns
                continue
            if ts_end_ns is not None and t_ns >= ts_end_ns:
                t_ns += dt_tau_ns
                continue

            t_next_ns = t_ns + dt_tau_ns
            lo = int(np.searchsorted(idx_ns, t_ns, side="left"))
            hi = int(np.searchsorted(idx_ns, t_next_ns, side="left"))

            if hi - lo == tau:
                hi_p = hi_arr[lo:hi].max()
                lo_p = lo_arr[lo:hi].min()
                op = op_arr[lo]

                t_list.append(pd.Timestamp(t_ns))
                ell_r.append(int(round((hi_p - lo_p) / tick)))
                ell_u.append(int(round(max(hi_p - op, 0.0) / tick)))
                ell_d.append(int(round(max(op - lo_p, 0.0) / tick)))
                vol_list.append(float(vol_arr[lo:hi].sum()))
                op_first_list.append(float(op))

            t_ns += dt_tau_ns

    # dx[i] = x[i+1] - x[i] where x = open of window i.
    # NaN whenever windows i and i+1 are not strictly adjacent (discontinuity).
    n = len(t_list)
    dx_list = [np.nan] * n
    for i in range(n - 1):
        if int(t_list[i + 1].value) - int(t_list[i].value) == dt_tau_ns:
            dx_list[i] = float(round((op_first_list[i + 1] - op_first_list[i]) / tick))

    return t_list, ell_r, ell_u, ell_d, vol_list, dx_list
