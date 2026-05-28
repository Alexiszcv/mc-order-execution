
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from order_mgmt.loader import (
    FULL_SESSION_QUANTILE,
    MIN_FRACTION_FOR_FULL_DAY,
    active_day_mask,
)
from order_mgmt.ticks import infer_tick


def _compute_stats(
    df: pd.DataFrame,
    *,
    min_fraction: float = MIN_FRACTION_FOR_FULL_DAY,
    full_session_quantile: float = FULL_SESSION_QUANTILE,
):
    """
    From a 1-minute bar DataFrame, return (daily, tick, proper_days, n_green, n_total).
    Called once per request so derived quantities are never recomputed from disk.
    """
    # Active mask is computed over trading days only; the full-session quantile must not
    # see the zero-filled calendar gaps added below for the figure.
    counts = df.groupby(df.index.normalize()).size().rename("traded_mins")
    active = active_day_mask(counts, min_fraction, full_session_quantile=full_session_quantile)

    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    daily = counts.to_frame().reindex(full_index, fill_value=0)
    daily["is_active"] = active.reindex(full_index, fill_value=False)

    prices = pd.concat([df["open"], df["high"], df["low"], df["close"]])
    tick = infer_tick(prices.to_numpy())

    proper_days = list(daily[daily["is_active"]].index)
    n_green     = int(daily["is_active"].sum())
    n_total     = len(daily)

    return daily, tick, proper_days, n_green, n_total


def _build_figure(df: pd.DataFrame, ticker: str, daily, max_traded: int):
    """Build the volume figure from pre-loaded data (no CSV read)."""
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.set_facecolor("#FDE8E6")           # pastel red for all days
    for date in daily.index[daily["is_active"]]:
        ax.axvspan(date, date + pd.Timedelta(days=1),
                   color="#4CAF50", alpha=0.12, linewidth=0)

    vol_daily = df["volume"].resample("D").sum()
    vol_daily = vol_daily[vol_daily > 0]
    t_num = mdates.date2num(vol_daily.index.to_pydatetime())
    ax.bar(t_num, vol_daily.to_numpy(), width=0.8,
           color="steelblue", alpha=0.8)
    ax.xaxis_date()

    ax.set_xlabel("Time")
    ax.set_ylabel("Volume")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.margins(x=0.01)
    fig.subplots_adjust(bottom=0.18, top=0.88)   # replaces tight_layout (much faster)
    return fig


def get_tick(csv_path: str) -> float:
    """Kept for standalone use outside the web app."""
    df = pd.read_csv(csv_path, header=None,
                     names=["time", "open", "high", "low", "close", "volume"])
    prices = pd.concat([df["open"], df["high"], df["low"], df["close"]])
    return infer_tick(prices.to_numpy())


def get_proper_days(csv_path: str) -> list:
    """Kept for standalone use outside the web app."""
    df = pd.read_csv(csv_path, header=None,
                     names=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d.%H:%M:%S")
    counts = df.groupby(df["time"].dt.normalize()).size()
    return list(counts[active_day_mask(counts)].index)
