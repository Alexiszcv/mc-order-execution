"""
Minimal localhost web interface.
Run: python src/app.py  →  http://localhost:8000
"""

import base64
import io
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sys.path.insert(0, str(Path(__file__).parent))
from plot_volume import _build_figure, _compute_stats  # noqa: E402
from epdf import _load_1min, build_epdf  # noqa: E402
from plotting import build_histogram_figure as _build_histogram_figure  # noqa: E402
from ranges import compute_all_ranges  # noqa: E402
from regime import compute_ewma_series, _build_regime_figure  # noqa: E402

from order_mgmt.backtest import run_backtest  # noqa: E402
from order_mgmt.baselines import vwap_baseline  # noqa: E402
from order_mgmt.ticks import resolve_tick  # noqa: E402

EPDF_J_START = 200
BACKTEST_FILL_RATE_TARGET = 0.6


def _build_backtest_figure(
    strat_buy_slip, vwap_buy_slip, strat_sell_slip, vwap_sell_slip, ticker: str
):
    """1x2 slippage histograms (buy / sell) — strategy vs VWAP."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    panels = [
        (axes[0], "buy", strat_buy_slip, vwap_buy_slip),
        (axes[1], "sell", strat_sell_slip, vwap_sell_slip),
    ]
    for ax, side, strat, vwap in panels:
        if strat:
            ax.hist(strat, bins=40, alpha=0.6, label="Strategy", color="steelblue")
        if vwap:
            ax.hist(vwap, bins=40, alpha=0.6, label="VWAP", color="orange")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_title(f"{ticker} — {side} (slippage vs open)")
        ax.set_xlabel("ticks")
        ax.set_ylabel("count")
        ax.legend()
    plt.tight_layout()
    return fig


def _build_epdf_summary(counts_RU, counts_RD, M: int, N: int, K: int) -> str:
    """One-row-per-cell HTML table of regime → (n_obs, mean R_U ticks, mean R_D ticks)."""
    rows = []
    rows.append(
        "<tr><th>(m,n,k)</th><th>n</th><th>mean R<sub>U</sub></th>"
        "<th>mean R<sub>D</sub></th></tr>"
    )
    for m in range(1, M + 1):
        for n_st in range(1, N + 1):
            for k in range(1, K + 1):
                ru = counts_RU.get((m, n_st, k))
                rd = counts_RD.get((m, n_st, k))
                if not ru:
                    continue
                n_obs = sum(ru.values())
                mean_ru = sum(ell * c for ell, c in ru.items()) / n_obs
                mean_rd = (
                    sum(ell * c for ell, c in rd.items()) / sum(rd.values()) if rd else 0.0
                )
                rows.append(
                    f"<tr><td>({m},{n_st},{k})</td>"
                    f"<td style='text-align:right'>{n_obs}</td>"
                    f"<td style='text-align:right'>{mean_ru:.2f}</td>"
                    f"<td style='text-align:right'>{mean_rd:.2f}</td></tr>"
                )
    if len(rows) == 1:
        return "<p style='color:#666'>No populated regime cells (try lowering j_start or fewer states).</p>"
    return (
        "<table style='border-collapse:collapse;font-size:.85rem'>"
        + "".join(rows)
        + "</table>"
    )


def _scan_contracts():
    return sorted(
        p for p in DATA.rglob("*.csv") if not p.stem.startswith("AIAgent")
    )


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _render(csv_path: Path, tau: int, half_life: int,
            n_states_range: int, n_states_vol: int, k_states_dx: int) -> dict:
    ticker  = csv_path.stem
    df_1min = _load_1min(str(csv_path))                          # single CSV read

    daily, tick, proper_days, n_green, n_total = _compute_stats(df_1min)
    max_traded = int(daily["traded_mins"].max())

    fig_vol  = _build_figure(df_1min, ticker, daily, max_traded)
    t_list, ell_r, ell_u, ell_d, vol_list, dx_list = compute_all_ranges(
        df_1min, tau, tick, proper_days
    )
    fig_hist = _build_histogram_figure(ell_r, ell_u, ell_d, tau, tick, ticker)

    ewma_range, ewma_vol = compute_ewma_series(t_list, ell_r, vol_list, half_life)
    fig_regime = _build_regime_figure(
        t_list, ewma_range, ewma_vol, dx_list, half_life, ticker,
        n_states_range=n_states_range, n_states_vol=n_states_vol,
        k_states_dx=k_states_dx,
    )

    counts_RU, counts_RD, _thr = build_epdf(
        t_list, ell_u, ell_d, list(ewma_vol), list(ewma_range), dx_list,
        M=n_states_vol, N=n_states_range, K=k_states_dx, j_start=EPDF_J_START,
    )
    epdf_table = _build_epdf_summary(
        counts_RU, counts_RD, M=n_states_vol, N=n_states_range, K=k_states_dx
    )

    # Backtest: regime-conditioned strategy vs VWAP, both sides.
    tick_eff = resolve_tick(ticker, tick)
    bt_buy = run_backtest(
        df_1min, tau=tau, tick=tick_eff, proper_days=proper_days, side="buy",
        fill_rate_target=BACKTEST_FILL_RATE_TARGET, half_life=half_life,
        M=n_states_vol, N=n_states_range, K=k_states_dx, j_start=EPDF_J_START,
    )
    bt_sell = run_backtest(
        df_1min, tau=tau, tick=tick_eff, proper_days=proper_days, side="sell",
        fill_rate_target=BACKTEST_FILL_RATE_TARGET, half_life=half_life,
        M=n_states_vol, N=n_states_range, K=k_states_dx, j_start=EPDF_J_START,
    )
    vwap_buy = vwap_baseline(df_1min, t_list[EPDF_J_START:], tau=tau, tick=tick_eff, side="buy")
    vwap_sell = vwap_baseline(df_1min, t_list[EPDF_J_START:], tau=tau, tick=tick_eff, side="sell")

    fig_bt = _build_backtest_figure(
        bt_buy.slippage_ticks, vwap_buy.slippage_ticks,
        bt_sell.slippage_ticks, vwap_sell.slippage_ticks,
        ticker,
    )

    return dict(
        vol_b64    = _fig_to_b64(fig_vol),
        hist_b64   = _fig_to_b64(fig_hist),
        regime_b64 = _fig_to_b64(fig_regime),
        bt_b64     = _fig_to_b64(fig_bt),
        epdf_table = epdf_table,
        n_green    = n_green,
        n_total    = n_total,
        tick       = tick_eff,
        n_windows  = len(ell_r),
        bt_buy     = bt_buy,
        bt_sell    = bt_sell,
        vwap_buy_avg  = float(np.mean(vwap_buy.slippage_ticks)) if vwap_buy.slippage_ticks else 0.0,
        vwap_sell_avg = float(np.mean(vwap_sell.slippage_ticks)) if vwap_sell.slippage_ticks else 0.0,
    )


TAU_VALUES = [1, 5, 10, 15, 30, 60]
HALF_LIFE_MIN, HALF_LIFE_MAX, HALF_LIFE_DEFAULT = 5, 200, 20
STATES_MIN, STATES_MAX, STATES_DEFAULT = 2, 6, 3


def _html(contracts, selected_stem="", tau=5, half_life=HALF_LIFE_DEFAULT,
          n_states_range=STATES_DEFAULT, n_states_vol=STATES_DEFAULT,
          k_states_dx=STATES_DEFAULT, data=None):
    contract_opts = "\n".join(
        f'<option value="{p.stem}" {"selected" if p.stem == selected_stem else ""}>'
        f'{p.stem}  ({p.parent.name})</option>'
        for p in contracts
    )

    tau = tau if tau in TAU_VALUES else TAU_VALUES[0]
    idx = TAU_VALUES.index(tau)
    hl  = max(HALF_LIFE_MIN, min(HALF_LIFE_MAX, half_life))
    nr  = max(STATES_MIN, min(STATES_MAX, n_states_range))
    nv  = max(STATES_MIN, min(STATES_MAX, n_states_vol))
    kd  = max(STATES_MIN, min(STATES_MAX, k_states_dx))

    def img(b64):
        return (f'<img src="data:image/png;base64,{b64}"'
                f' style="max-width:100%;margin-top:20px">') if b64 else ""

    stats = ""
    vol_section = hist_section = regime_section = epdf_section = backtest_section = ""
    if data:
        d = data
        stats = (
            f'<p style="margin-top:14px">'
            f'<strong>{d["n_green"]}</strong> active days'
            f' / <strong>{d["n_total"]}</strong> total'
            f' ({100*d["n_green"]/d["n_total"]:.1f} %)'
            f' &emsp;|&emsp; tick = <strong>{d["tick"]:g}</strong>'
            f' &emsp;|&emsp; <strong>{d["n_windows"]}</strong> valid windows'
            f' (τ = {tau} min)'
            f'</p>'
        )
        vol_section = (
            '<h3 style="margin-top:32px;border-top:1px solid #ddd;padding-top:14px">'
            'Daily traded volume</h3>'
            f'<p style="color:#666;font-size:.9rem">Green days: traded_mins ≥ 90 % of maximum across the dataset.</p>'
            + img(d["vol_b64"])
        )
        hist_section = (
            '<h3 style="margin-top:32px;border-top:1px solid #ddd;padding-top:14px">'
            'Intra-window range distributions</h3>'
            f'<p style="color:#666;font-size:.9rem">'
            f'R = high − low &emsp;|&emsp; R<sub>U</sub> = high − open &emsp;|&emsp; '
            f'R<sub>D</sub> = open − low &emsp; (in ticks, τ = {tau} min)</p>'
            + img(d["hist_b64"])
        )
        regime_section = (
            '<h3 style="margin-top:32px;border-top:1px solid #ddd;padding-top:14px">'
            'Regime indicators</h3>'
            f'<p style="color:#666;font-size:.9rem">'
            f'EWMA Range and Volume (half-life = {half_life} windows) and '
            f'Δx = open[t+τ] − open[t] (in ticks). '
            f'Colour bands = quantile-based states.</p>'
            + img(d["regime_b64"])
        )
        epdf_section = (
            '<h3 style="margin-top:32px;border-top:1px solid #ddd;padding-top:14px">'
            'Conditional ePDFs (regime → R<sub>U</sub>/R<sub>D</sub> ticks)</h3>'
            f'<p style="color:#666;font-size:.9rem">'
            f'Per-regime cell (m=volume, n=range, k=direction): count of windows and '
            f'mean R<sub>U</sub>/R<sub>D</sub> in ticks. Skips j&lt;{EPDF_J_START} '
            f'(warm-up).</p>'
            + d["epdf_table"]
        )
        bb, bs = d["bt_buy"], d["bt_sell"]
        backtest_section = (
            '<h3 style="margin-top:32px;border-top:1px solid #ddd;padding-top:14px">'
            'Backtest — strategy vs VWAP (TWAP=open is the zero baseline)</h3>'
            f'<p style="color:#666;font-size:.9rem">'
            f'fill_rate_target = {BACKTEST_FILL_RATE_TARGET}. '
            f'<strong>buy:</strong> n={bb.n_decisions}, fill={bb.fill_rate:.1%}, '
            f'avg={bb.avg_slippage_ticks:+.2f}t, median={bb.median_slippage_ticks:+.2f}t '
            f'(VWAP avg={d["vwap_buy_avg"]:+.2f}t). '
            f'<strong>sell:</strong> n={bs.n_decisions}, fill={bs.fill_rate:.1%}, '
            f'avg={bs.avg_slippage_ticks:+.2f}t, median={bs.median_slippage_ticks:+.2f}t '
            f'(VWAP avg={d["vwap_sell_avg"]:+.2f}t).</p>'
            + img(d["bt_b64"])
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Futures Execution Explorer</title>
  <style>
    body  {{ font-family: sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 16px; }}
    select, button {{ font-size: 1rem; padding: 6px 10px; }}
    button {{ margin-left: 10px; cursor: pointer; }}
    .ctrl-row {{ display:flex; align-items:center; gap:14px; margin-top:10px; flex-wrap:wrap; }}
    input[type=range] {{ width: 160px; }}
    label {{ font-size: 1rem; }}
    .sep {{ color:#ccc; }}
    h3 {{ font-size:1.05rem; color:#333; margin-bottom:4px; }}
  </style>
</head>
<body>
  <h2>Futures Execution Explorer</h2>
  <form method="get" action="/plot">
    <select name="contract">{contract_opts}</select>

    <div class="ctrl-row">
      <label>τ = <strong id="tv">{tau}</strong> min</label>
      <input type="range" min="0" max="5" step="1" value="{idx}"
             oninput="var v=[1,5,10,15,30,60][+this.value];
                      document.getElementById('tv').textContent=v;
                      document.getElementById('tau_h').value=v;">
      <input type="hidden" id="tau_h" name="tau" value="{tau}">

      <span class="sep">|</span>

      <label>half-life = <strong id="hlv">{hl}</strong> windows</label>
      <input type="range" min="{HALF_LIFE_MIN}" max="{HALF_LIFE_MAX}" step="1" value="{hl}"
             oninput="document.getElementById('hlv').textContent=this.value;
                      document.getElementById('hl_h').value=this.value;">
      <input type="hidden" id="hl_h" name="half_life" value="{hl}">
    </div>

    <div class="ctrl-row">
      <label>N states (range) = <strong id="nrv">{nr}</strong></label>
      <input type="range" min="{STATES_MIN}" max="{STATES_MAX}" step="1" value="{nr}"
             oninput="document.getElementById('nrv').textContent=this.value;
                      document.getElementById('nr_h').value=this.value;">
      <input type="hidden" id="nr_h" name="n_range" value="{nr}">

      <span class="sep">|</span>

      <label>M states (volume) = <strong id="nvv">{nv}</strong></label>
      <input type="range" min="{STATES_MIN}" max="{STATES_MAX}" step="1" value="{nv}"
             oninput="document.getElementById('nvv').textContent=this.value;
                      document.getElementById('nv_h').value=this.value;">
      <input type="hidden" id="nv_h" name="n_vol" value="{nv}">

      <span class="sep">|</span>

      <label>K states (Δx) = <strong id="kdv">{kd}</strong></label>
      <input type="range" min="{STATES_MIN}" max="{STATES_MAX}" step="1" value="{kd}"
             oninput="document.getElementById('kdv').textContent=this.value;
                      document.getElementById('kd_h').value=this.value;">
      <input type="hidden" id="kd_h" name="k_dx" value="{kd}">

      <button type="submit">Plot</button>
    </div>
  </form>
  {stats}
  {vol_section}
  {hist_section}
  {regime_section}
  {epdf_section}
  {backtest_section}
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    contracts = _scan_contracts()

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: str, status=200):
        encoded = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send(_html(self.contracts))

        elif parsed.path == "/plot":
            params    = parse_qs(parsed.query)
            stem      = params.get("contract", [""])[0]
            tau_raw   = int(params.get("tau",        [5])[0])
            tau       = tau_raw if tau_raw in TAU_VALUES else TAU_VALUES[0]
            hl_raw    = int(params.get("half_life",  [HALF_LIFE_DEFAULT])[0])
            half_life = max(HALF_LIFE_MIN, min(HALF_LIFE_MAX, hl_raw))
            nr_raw    = int(params.get("n_range",    [STATES_DEFAULT])[0])
            n_range   = max(STATES_MIN, min(STATES_MAX, nr_raw))
            nv_raw    = int(params.get("n_vol",      [STATES_DEFAULT])[0])
            n_vol     = max(STATES_MIN, min(STATES_MAX, nv_raw))
            kd_raw    = int(params.get("k_dx",       [STATES_DEFAULT])[0])
            k_dx      = max(STATES_MIN, min(STATES_MAX, kd_raw))

            match = next((p for p in self.contracts if p.stem == stem), None)
            if match is None:
                self._send("<p>Contract not found.</p>", 404)
                return
            data = _render(match, tau, half_life, n_range, n_vol, k_dx)
            self._send(_html(self.contracts, selected_stem=stem, tau=tau,
                             half_life=half_life, n_states_range=n_range,
                             n_states_vol=n_vol, k_states_dx=k_dx, data=data))

        else:
            self._send("<p>Not found.</p>", 404)


if __name__ == "__main__":
    port = 8000
    server = HTTPServer(("localhost", port), Handler)
    print(f"Listening on http://localhost:{port}")
    server.serve_forever()
