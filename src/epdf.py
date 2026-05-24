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

    t_list, value_list, ell_list = [], [], []

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

            t += dt_tau

    return t_list, value_list, ell_list


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
    t_list, ell_r, ell_u, ell_d  — four parallel lists
    """
    NS_PER_DAY  = 86_400 * 10 ** 9
    dt_tau_ns   = int(pd.Timedelta(minutes=tau).value)
    dt_last_ns  = int(pd.Timedelta(minutes=tau - 1).value)
    ts_start_ns = int(pd.Timestamp(t_start).value) if t_start is not None else None
    ts_end_ns   = int(pd.Timestamp(t_end).value)   if t_end   is not None else None

    # Pre-group: one O(n_total) pass to build per-day numpy arrays
    proper_set = {pd.Timestamp(d).normalize() for d in proper_days_list}
    day_data   = {}                          # day_ts → (idx_ns, high, low, open)
    for day_ts, grp in df_1min.groupby(df_1min.index.normalize()):
        if day_ts in proper_set:
            day_data[day_ts] = (
                grp.index.asi8,              # int64 ns — sortable with np.searchsorted
                grp["high"].to_numpy(),
                grp["low"].to_numpy(),
                grp["open"].to_numpy(),
            )

    t_list, ell_r, ell_u, ell_d = [], [], [], []

    for day in proper_days_list:
        day_ts = pd.Timestamp(day).normalize()
        if day_ts not in day_data:
            continue
        idx_ns, hi_arr, lo_arr, op_arr = day_data[day_ts]

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

            t_ns += dt_tau_ns

    return t_list, ell_r, ell_u, ell_d


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

    fig.suptitle(
        f"{ticker} — distributions of R / R_U / R_D  (τ = {tau} min, tick = {tick:g})",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


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
