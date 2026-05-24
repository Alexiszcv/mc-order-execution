import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def _build_figure(csv_path: str):
    df = pd.read_csv(csv_path, header=None,
                     names=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d.%H:%M:%S")
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.normalize()

    daily = (
        df.groupby("date")
        .agg(traded_mins=("time", "count"))
        .reset_index()
        .set_index("date")
    )

    # Fill every calendar day between first and last, gaps get traded_mins = 0
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0)
    daily.index.name = "date"

    max_traded = daily["traded_mins"].max()
    daily["is_active"] = daily["traded_mins"] >= 0.90 * max_traded

    fig, ax = plt.subplots(figsize=(15, 5))

    for date, row in daily.iterrows():
        color = "#4CAF50" if row["is_active"] else "#F44336"
        ax.axvspan(date, date + pd.Timedelta(days=1),
                   color=color, alpha=0.12, linewidth=0)

    ax.vlines(df["time"], 0, df["volume"],
              color="steelblue", linewidth=0.8, alpha=0.7)
    ax.plot(df["time"], df["volume"],
            "o", color="steelblue", markersize=2, alpha=0.6)

    ticker = Path(csv_path).stem
    ax.set_title(f"{ticker} — Volume per minute\n"
                 f"(green: traded_mins ≥ 90 % of max = {max_traded})",
                 fontsize=11)
    ax.set_xlabel("Time")
    ax.set_ylabel("Volume")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.margins(x=0.01)
    plt.tight_layout()
    n_green = int(daily["is_active"].sum())
    n_total = len(daily)   # all calendar days first→last, including gaps
    return fig, n_green, n_total


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
