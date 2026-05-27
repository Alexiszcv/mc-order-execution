import bisect
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def compute_ranges(df_1min: pd.DataFrame, tau: int, tick: float,
                   proper_days_list, quantity: str,
                   t_start=None, t_end=None):
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

    dt_tau  = pd.Timedelta(minutes=tau)
    dt_last = pd.Timedelta(minutes=tau - 1)   # last minute inside the window

    t_list, value_list, ell_list, vol_list = [], [], [], []

    for day in proper_days_list:
        day_ts   = pd.Timestamp(day).normalize()
        day_bars = df_1min[df_1min.index.normalize() == day_ts]
        if day_bars.empty:
            continue

        t = day_bars.index[0]           # start of first window

        while t <= day_bars.index[-1]:
            # window crosses midnight → stop for this day
            if (t + dt_last).date() != t.date():
                break

            # optional time filter
            if t_start is not None and t < pd.Timestamp(t_start):
                t += dt_tau
                continue
            if t_end is not None and t >= pd.Timestamp(t_end):
                t += dt_tau
                continue

            t_next  = t + dt_tau
            mask    = (day_bars.index >= t) & (day_bars.index < t_next)
            window  = day_bars.loc[mask]

            if len(window) == tau:
                hi         = window["high"].max()
                lo         = window["low"].min()
                first_open = window["open"].iloc[0]

                if quantity == "R":
                    value = hi - lo
                elif quantity == "R_U":
                    value = hi - first_open
                else:                    # R_D
                    value = first_open - lo

                t_list.append(t)
                value_list.append(float(value))
                ell_list.append(int(round(value / tick)))
                vol_list.append(float(window["volume"].sum()))

            t += dt_tau

    return t_list, value_list, ell_list, vol_list


def compute_all_ranges(df_1min: pd.DataFrame, tau: int, tick: float,
                       proper_days_list, t_start=None, t_end=None):
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
    NS_PER_DAY  = 86_400 * 10 ** 9
    dt_tau_ns   = int(pd.Timedelta(minutes=tau).value)
    dt_last_ns  = int(pd.Timedelta(minutes=tau - 1).value)
    ts_start_ns = int(pd.Timestamp(t_start).value) if t_start is not None else None
    ts_end_ns   = int(pd.Timestamp(t_end).value)   if t_end   is not None else None

    # Pre-group: one O(n_total) pass to build per-day numpy arrays.
    # Force-convert each group's index to ns so the int64 arithmetic below
    # is unit-correct on pandas 3.0+ (where the default datetime unit is us, not ns).
    proper_set = {pd.Timestamp(d).normalize() for d in proper_days_list}
    day_data = {}  # day_ts → (idx_ns, high, low, open, vol)
    for day_ts, grp in df_1min.groupby(df_1min.index.normalize()):
        if day_ts in proper_set:
            day_data[day_ts] = (
                grp.index.as_unit("ns").asi8,  # int64 ns — sortable with np.searchsorted
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

        t_ns   = int(idx_ns[0])
        end_ns = int(idx_ns[-1])

        while t_ns <= end_ns:
            # stop if the last minute of the window would cross midnight
            if (t_ns + dt_last_ns) // NS_PER_DAY != t_ns // NS_PER_DAY:
                break
            if ts_start_ns is not None and t_ns <  ts_start_ns:
                t_ns += dt_tau_ns; continue
            if ts_end_ns   is not None and t_ns >= ts_end_ns:
                t_ns += dt_tau_ns; continue

            t_next_ns = t_ns + dt_tau_ns
            lo = int(np.searchsorted(idx_ns, t_ns,      side="left"))
            hi = int(np.searchsorted(idx_ns, t_next_ns, side="left"))

            if hi - lo == tau:
                hi_p = hi_arr[lo:hi].max()
                lo_p = lo_arr[lo:hi].min()
                op   = op_arr[lo]

                t_list.append(pd.Timestamp(t_ns))
                ell_r.append(int(round((hi_p - lo_p)         / tick)))
                ell_u.append(int(round(max(hi_p - op,   0.0) / tick)))
                ell_d.append(int(round(max(op   - lo_p, 0.0) / tick)))
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


def _build_histogram_figure(ell_r, ell_u, ell_d, tau: int, tick: float, ticker: str):
    """Three side-by-side integer histograms of R, R_U, R_D in ticks."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        (ell_r, "R",   "steelblue"),
        (ell_u, "R_U", "seagreen"),
        (ell_d, "R_D", "tomato"),
    ]
    for ax, (data, label, color) in zip(axes, specs):
        if data:
            p99 = int(np.percentile(data, 99))
            bins = range(0, p99 + 2)
            ax.set_xlim(0, p99 + 1)
        else:
            bins = 10
        ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("ticks (ℓ)")
        ax.set_ylabel("count")
        ax.margins(x=0.02)

    plt.tight_layout()
    return fig


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
    K            : number of Δx    states  {1..K}
    j_start      : first index to start accumulating (must be >= 1)

    Returns
    -------
    counts_RU  : dict {(m, n, k): Counter}   keyed by (vol_state, range_state, dx_state)
    counts_RD  : dict {(m, n, k): Counter}
    thresholds : dict {'vol': list, 'range': list, 'dx': list}
                 Final quantile thresholds over the full series (for display).
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

    # Final quantile thresholds on the full series (for display in the interface).
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
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from plot_volume import get_tick

    csv = Path(__file__).parents[1] / "data" / "Gold" / "GCG24.csv"
    tick = get_tick(str(csv))

    df_1min = _load_1min(str(csv))

    df_tau = resample_to_tau(df_1min, tau=5)
    df_out = compute_spreads(df_tau, tick)

    ok = (df_out["R"] == df_out["R_U"] + df_out["R_D"]).all()
    print(f"tick      = {tick:g}")
    print(f"tau bars  = {len(df_out)}")
    print(f"R == R_U + R_D : {ok}")
    print(df_out[["open", "high", "low", "close", "volume", "R", "R_U", "R_D"]].head(10))
    if not ok:
        bad = df_out[df_out["R"] != df_out["R_U"] + df_out["R_D"]]
        print(f"\n{len(bad)} mismatches (rounding artefacts):")
        print(bad[["R", "R_U", "R_D"]].head())

    # --- test compute_ranges ---
    print("\n--- compute_ranges ---")
    TAU = 5
    daily_counts = df_1min.groupby(df_1min.index.normalize()).size()
    max_traded   = daily_counts.max()
    proper_days  = list(daily_counts[daily_counts >= 0.9 * max_traded].index)
    print(f"proper days : {len(proper_days)}")

    for qty in ("R", "R_U", "R_D"):
        t, v, ell = compute_ranges(df_1min, TAU, tick, proper_days, qty)
        print(f"  {qty}: {len(t)} windows, ell in [{min(ell)}, {max(ell)}]")

    # cross-check: R_ell == R_U_ell + R_D_ell on shared windows
    t_r,  _,  ell_r  = compute_ranges(df_1min, TAU, tick, proper_days, "R")
    t_u,  _,  ell_u  = compute_ranges(df_1min, TAU, tick, proper_days, "R_U")
    t_d,  _,  ell_d  = compute_ranges(df_1min, TAU, tick, proper_days, "R_D")
    assert t_r == t_u == t_d, "timestamp lists differ"
    mismatches = sum(r != u + d for r, u, d in zip(ell_r, ell_u, ell_d))
    print(f"  ell_R == ell_U + ell_D mismatches (rounding): {mismatches}/{len(t_r)}")
