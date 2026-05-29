import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def ewma_ewmv(eta, half_life: int):
    """
    Algorithm 1 — EWMA / EWMV on series eta with half-life m = half_life.

    lam = 2^(-1/m)

    j=1: all accumulators initialised to 0, output NaN
    j=2: sumW=1, sumWX=eta[0], ewma=sumWX/sumW,
         sumWSS=(eta[0]-ewma)², ewmv=sqrt(sumWSS/sumW), output NaN
    j≥3: sumW = lam·sumW + 1
         sumWX = lam·sumWX + eta[j-1]
         ewma  = sumWX / sumW
         sumWSS = lam·sumWSS + (eta[j-1] - ewma)²
         ewmv  = sqrt(sumWSS / sumW)
         output written at position j-1

    Returns two numpy arrays of length n, with NaN at positions 0 and 1.

    Note (EWMV bias): the deviation term uses the *current* mean,
    ``(eta - ewma)`` with ``ewma`` already updated by the new observation. This
    is faithful to Algorithm 1 (line 20) but differs from EWMV variants that use
    the prior mean; it imparts a small downward bias. Flagged for the team;
    unchanged.
    """
    eta = np.asarray(eta, dtype=float)
    n   = len(eta)
    lam = 2.0 ** (-1.0 / half_life)

    out_ewma = np.full(n, np.nan)
    out_ewmv = np.full(n, np.nan)

    if n < 3:
        return out_ewma, out_ewmv

    # j = 1: init all accumulators to 0
    sumW = sumWX = sumWSS = 0.0

    # j = 2: first observation loaded
    sumW   = 1.0
    sumWX  = eta[0]
    ewma_v = sumWX / sumW
    sumWSS = (eta[0] - ewma_v) ** 2
    ewmv_v = (sumWSS / sumW) ** 0.5
    # positions 0 and 1 stay NaN

    # SPEC DEVIATION (do not change without team sign-off): Algorithm 1 folds in
    # the prior interval eta[j-2] (eta_{j-1} in the spec's 1-based j) at step j;
    # this loop reads eta[j-1], so eta[1] is never used and the series leads the
    # spec by one observation. Pinned by tests/test_ewma_spec.py; see
    # notes/component-review.md ("Stream A findings").
    for j in range(3, n + 1):
        x      = eta[j - 1]
        sumW   = lam * sumW   + 1.0
        sumWX  = lam * sumWX  + x
        ewma_v = sumWX / sumW
        sumWSS = lam * sumWSS + (x - ewma_v) ** 2
        ewmv_v = (sumWSS / sumW) ** 0.5
        out_ewma[j - 1] = ewma_v
        out_ewmv[j - 1] = ewmv_v

    return out_ewma, out_ewmv


def compute_ewma_series(t_list, value_list, vol_list, half_life: int):
    """
    Apply ewma_ewmv to value_list (range) and vol_list (volume).
    Returns (ewma_range, ewma_vol), each a numpy array of length len(t_list),
    with NaN at the first two positions.
    """
    ewma_range, _ = ewma_ewmv(value_list, half_life)
    ewma_vol,   _ = ewma_ewmv(vol_list,   half_life)
    return ewma_range, ewma_vol


_CMAP = plt.cm.RdYlGn


def _colored_panel(ax_ts, ax_hist, t_arr, data, n_states, label):
    """
    Fill ax_ts (time series) and ax_hist (horizontal histogram) with quantile-
    based colored horizontal bands, then draw the series line and histogram bars.
    Both axes must already share their y-axis.
    """
    arr = np.asarray(data, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return

    qs     = np.percentile(valid, [100.0 * k / n_states for k in range(1, n_states)])
    bounds = np.concatenate([[-np.inf], qs, [np.inf]])
    colors = _CMAP(np.linspace(0.0, 1.0, n_states))

    ymin, ymax = float(valid.min()), float(valid.max())
    pad  = max((ymax - ymin) * 0.05, 1e-9)
    y_lo, y_hi = ymin - pad, ymax + pad

    for k in range(n_states):
        lo = float(max(bounds[k],     y_lo))
        hi = float(min(bounds[k + 1], y_hi))
        ax_ts.axhspan(lo, hi,   color=colors[k], alpha=0.22, linewidth=0)
        ax_hist.axhspan(lo, hi, color=colors[k], alpha=0.22, linewidth=0)

    mask = ~np.isnan(arr)
    ax_ts.plot(
        [t_arr[i] for i in range(len(t_arr)) if mask[i]],
        arr[mask], color="black", linewidth=0.7,
    )
    ax_ts.set_ylim(y_lo, y_hi)
    ax_ts.set_title(label, fontsize=10)
    ax_ts.set_xlabel("Time")
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_ts.tick_params(axis="x", labelrotation=30)
    for lbl in ax_ts.get_xticklabels():
        lbl.set_ha("right")

    n_bins = min(80, max(20, len(valid) // 30))
    ax_hist.hist(valid, bins=n_bins, orientation="horizontal",
                 color="slategray", alpha=0.75, edgecolor="none")
    ax_hist.yaxis.set_visible(False)
    ax_hist.set_xlabel("count", fontsize=9)


def _build_regime_figure(t_list, ewma_range, ewma_vol, dx_list,
                         half_life: int, ticker: str,
                         n_states_range: int = 3, n_states_vol: int = 3,
                         k_states_dx: int = 3):
    """
    6-panel figure (3 rows × 2 cols): [time series | histogram] for each of
    EWMA Range, EWMA Volume, Δx.  Each row shares its y-axis.
    """
    t_arr = [t.to_pydatetime() for t in t_list]

    fig = plt.figure(figsize=(18, 12))
    gs  = fig.add_gridspec(3, 2, width_ratios=[4, 1], hspace=0.52, wspace=0.04)

    ax_tr = fig.add_subplot(gs[0, 0])
    ax_hr = fig.add_subplot(gs[0, 1], sharey=ax_tr)
    ax_tv = fig.add_subplot(gs[1, 0])
    ax_hv = fig.add_subplot(gs[1, 1], sharey=ax_tv)
    ax_td = fig.add_subplot(gs[2, 0])
    ax_hd = fig.add_subplot(gs[2, 1], sharey=ax_td)

    _colored_panel(ax_tr, ax_hr, t_arr, ewma_range, n_states_range,
                   f"EWMA Range  ({n_states_range} states, half-life={half_life})")
    _colored_panel(ax_tv, ax_hv, t_arr, ewma_vol,   n_states_vol,
                   f"EWMA Volume  ({n_states_vol} states, half-life={half_life})")
    _colored_panel(ax_td, ax_hd, t_arr, np.asarray(dx_list, dtype=float), k_states_dx,
                   f"Δx  ({k_states_dx} states)")

    return fig


if __name__ == "__main__":
    rng  = np.random.default_rng(0)
    eta  = rng.standard_normal(200)
    ewma, ewmv = ewma_ewmv(eta, half_life=20)

    assert np.isnan(ewma[0]) and np.isnan(ewma[1]), "positions 0 and 1 must be NaN"
    assert np.isnan(ewmv[0]) and np.isnan(ewmv[1])
    assert not np.any(np.isnan(ewma[2:])),          "no NaN beyond position 1"
    assert np.all(ewmv[2:] >= 0),                   "ewmv must be non-negative"

    print(f"n={len(eta)}, half_life=20, lam={2**(-1/20):.6f}")
    print(f"ewma[2:7]  = {ewma[2:7].round(4)}")
    print(f"ewmv[2:7]  = {ewmv[2:7].round(4)}")
    print("OK")
