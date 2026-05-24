import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def _compute_stats(df: pd.DataFrame):
    """
    From a 1-minute bar DataFrame, return (daily, tick, proper_days, n_green, n_total).
    Called once per request so derived quantities are never recomputed from disk.
    """
    daily = (
        df.groupby(df.index.normalize())
        .size()
        .rename("traded_mins")
        .to_frame()
    )
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0)

    max_traded = daily["traded_mins"].max()
    daily["is_active"] = daily["traded_mins"] >= 0.90 * max_traded

    prices = pd.concat([df["open"], df["high"], df["low"], df["close"]])
    prices = prices.round(8).drop_duplicates().sort_values().values
    diffs  = prices[1:] - prices[:-1]
    tick   = float(diffs[diffs > 1e-9].min())
    from math import log10, floor
    mag  = floor(log10(tick))
    tick = round(tick, -mag + 7)

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

    # Aggregate to 15-min buckets for display (30 k → ~2 k points, same visual)
    vol_display = df["volume"].resample("15min").sum()
    vol_display = vol_display[vol_display > 0]
    t_num = mdates.date2num(vol_display.index.to_pydatetime())
    ax.vlines(t_num, 0, vol_display.to_numpy(),
              color="steelblue", linewidth=1.2, alpha=0.8)
    ax.xaxis_date()

    ax.set_title(f"{ticker} — Volume per minute\n"
                 f"(green: traded_mins ≥ 90 % of max = {max_traded})",
                 fontsize=11)
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
    prices = prices.round(8).drop_duplicates().sort_values().values
    diffs = prices[1:] - prices[:-1]
    positive = diffs[diffs > 1e-9]
    raw = float(positive.min())
    from math import log10, floor
    mag = floor(log10(raw))
    return round(raw, -mag + 7)


def get_proper_days(csv_path: str) -> list:
    """Kept for standalone use outside the web app."""
    df = pd.read_csv(csv_path, header=None,
                     names=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d.%H:%M:%S")
    df["date"] = df["time"].dt.normalize()
    daily = df.groupby("date").agg(traded_mins=("time", "count"))
    max_traded = daily["traded_mins"].max()
    return list(daily[daily["traded_mins"] >= 0.90 * max_traded].index)


def plot_volume(csv_path: str) -> None:
    fig, n_green, n_total = _build_figure(csv_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_volume.py <path/to/file.csv>")
        sys.exit(1)
    plot_volume(sys.argv[1])
