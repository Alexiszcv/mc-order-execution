"""Diagnostic: are the big chase tails real 5-min moves, or session/overnight artifacts?

For each agent decision it reconstructs the execution window's actual 1-min bars and flags
whether the window crosses midnight, sits at a session edge, is thin (few bars / big internal
gaps), or is a genuine contiguous fast move. Prints the worst-shortfall windows + summaries.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from order_mgmt.agent.loader import (  # noqa: E402
    find_agent_csv,
    find_market_dir,
    load_agent_series,
    load_market_for_agent,
)
from order_mgmt.agent.metrics import _fit_regime_model, _window_bars  # noqa: E402
from order_mgmt.backtest import _state  # noqa: E402

TAU, FILL_RATE_TARGET, HALF_LIFE, J_START = 5, 0.6, 20, 200
M, N, K = 3, 3, 3


def main(substring: str = "Gold") -> None:
    from order_mgmt.agent.slicing import fill_single
    from order_mgmt.strategy import pick_ell_star
    from ranges import compute_all_ranges
    from regime import compute_ewma_series

    market_dir = find_market_dir(ROOT / "data", substring)
    df, tick, proper_days, _ = load_market_for_agent(market_dir)
    agent = load_agent_series(find_agent_csv(market_dir), market=substring)
    trades = agent.df[agent.df["side"].notna()]

    t_list, ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(df, TAU, tick, proper_days)
    ewma_range, ewma_vol = compute_ewma_series(t_list, ell_r, vol_list, HALF_LIFE)
    ev, er, dx = (np.asarray(a, float) for a in (ewma_vol, ewma_range, dx_list))
    t_ns_arr = np.asarray([pd.Timestamp(t).value for t in t_list], np.int64)
    train_count = int(np.searchsorted(t_ns_arr, pd.Timestamp(trades.index.min()).value, "left"))
    cRU, cRD, sv, sr, sd = _fit_regime_model(ell_u, ell_d, ev, er, dx, train_count, M, N, K, J_START)

    idx = df.index
    idx_ns = idx.as_unit("ns").asi8
    times = idx.to_numpy()
    o_, h_, l_, c_ = (df[col].to_numpy() for col in ("open", "high", "low", "close"))
    tau_ns = int(pd.Timedelta(minutes=TAU).value)

    rows = []
    for t, row in trades.iterrows():
        side = row["side"]
        arrival = float(row["price"])
        t_ns = int(pd.Timestamp(t).value)
        w = int(np.searchsorted(t_ns_arr, t_ns - tau_ns, "right")) - 1
        if w < 1 or np.isnan(ev[w]) or np.isnan(er[w]) or np.isnan(dx[w]):
            continue
        m = _state(float(ev[w]), sv, M)
        n = _state(float(er[w]), sr, N)
        k = _state(float(dx[w]), sd, K)
        if None in (m, n, k):
            continue
        epdf = (cRU if side == "sell" else cRD).get((m, n, k))
        if not epdf:
            continue
        ell = pick_ell_star(epdf, FILL_RATE_TARGET)
        lo = int(np.searchsorted(idx_ns, t_ns, "left"))
        hi = int(np.searchsorted(idx_ns, t_ns + tau_ns, "left"))
        if hi <= lo:
            continue
        bars = _window_bars(idx_ns, o_, h_, l_, c_, t_ns, tau_ns)
        o, h, low, c = bars
        price, _ = fill_single(side, ell, o, h, low, c, tick)
        short = (price - arrival) / tick if side == "sell" else (arrival - price) / tick

        bt = pd.DatetimeIndex(times[lo:hi])
        span_min = (bt[-1] - bt[0]) / pd.Timedelta(minutes=1)
        gaps = np.diff(bt.values).astype("timedelta64[m]").astype(int) if len(bt) > 1 else [0]
        rows.append({
            "t": t, "side": side, "short": short, "ell": ell, "n_bars": hi - lo,
            "span_min": span_min, "max_gap_min": int(max(gaps)) if len(gaps) else 0,
            "cross_midnight": bt[0].date() != bt[-1].date(),
            "hour": pd.Timestamp(t).hour, "jump_ticks": (float(c[-1]) - float(o[0])) / tick,
            "first_bar": bt[0], "last_bar": bt[-1],
            "arrival": arrival, "open": float(o[0]), "tick": tick, "fill": price,
        })

    d = pd.DataFrame(rows)
    print(f"\n=== {substring}: {len(d)} decisions ===")
    print(f"cross-midnight windows: {int(d['cross_midnight'].sum())} / {len(d)}")
    print(f"windows with an internal gap >1min: {int((d['max_gap_min'] > 1).sum())} / {len(d)}")
    print(f"n_bars distribution: {dict(Counter(d['n_bars']))}")

    worst = d.nsmallest(15, "short")
    print("\n--- 15 worst-shortfall windows (arrival = agent price, open = OHLC) ---")
    print(f"{'decision':19} {'short':>7} {'arrival':>9} {'open':>9} {'arr-open(t)':>11} {'jumpT':>7}")
    for _, r in worst.iterrows():
        basis = (r["arrival"] - r["open"]) / r["tick"]
        print(f"{r['t']!s:19} {r['short']:>+7.1f} {r['arrival']:>9.2f} {r['open']:>9.2f} "
              f"{basis:>+11.1f} {r['jump_ticks']:>+7.1f}")

    # Basis-immune metric: execution vs the OHLC window OPEN (same instrument as the fill).
    d["slip_vs_open"] = np.where(
        d["side"] == "sell", (d["fill"] - d["open"]) / d["tick"], (d["open"] - d["fill"]) / d["tick"]
    )
    print("\n--- shortfall vs ARRIVAL (agent price, basis-contaminated) vs slippage vs OPEN (basis-immune) ---")
    for col, name in (("short", "vs arrival"), ("slip_vs_open", "vs open")):
        s = d[col]
        print(f"  {name:12} mean={s.mean():+6.2f}  median={s.median():+6.2f}  "
              f"p5={np.percentile(s, 5):+7.1f}  p1={np.percentile(s, 1):+7.1f}")

    big = d[d["short"] < -20]
    print(f"\ntails worse than -20t: {len(big)}; their hours: {dict(Counter(big['hour']))}")
    print(f"of those, cross-midnight: {int(big['cross_midnight'].sum())}, "
          f"internal-gap>1min: {int((big['max_gap_min'] > 1).sum())}, "
          f"n_bars<{TAU}: {int((big['n_bars'] < TAU).sum())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Gold")
