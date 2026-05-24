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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sys.path.insert(0, str(Path(__file__).parent))
from plot_volume import _build_figure, _compute_stats  # noqa: E402
from epdf import _load_1min, compute_all_ranges, _build_histogram_figure  # noqa: E402


def _scan_contracts():
    return sorted(
        p for p in DATA.rglob("*.csv") if not p.stem.startswith("AIAgent")
    )


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _render(csv_path: Path, tau: int) -> dict:
    ticker  = csv_path.stem
    df_1min = _load_1min(str(csv_path))                          # single CSV read

    daily, tick, proper_days, n_green, n_total = _compute_stats(df_1min)
    max_traded = int(daily["traded_mins"].max())

    fig_vol  = _build_figure(df_1min, ticker, daily, max_traded)
    _, ell_r, ell_u, ell_d = compute_all_ranges(df_1min, tau, tick, proper_days)
    fig_hist = _build_histogram_figure(ell_r, ell_u, ell_d, tau, tick, ticker)

    return dict(
        vol_b64   = _fig_to_b64(fig_vol),
        hist_b64  = _fig_to_b64(fig_hist),
        n_green   = n_green,
        n_total   = n_total,
        tick      = tick,
        n_windows = len(ell_r),
    )


TAU_VALUES = [1, 5, 10, 15, 30, 60]


def _html(contracts, selected_stem="", tau=5, data=None):
    contract_opts = "\n".join(
        f'<option value="{p.stem}" {"selected" if p.stem == selected_stem else ""}>'
        f'{p.stem}  ({p.parent.name})</option>'
        for p in contracts
    )

    tau   = tau if tau in TAU_VALUES else TAU_VALUES[0]
    idx   = TAU_VALUES.index(tau)       # slider position (0–4)

    def img(b64):
        return (f'<img src="data:image/png;base64,{b64}"'
                f' style="max-width:100%;margin-top:20px">') if b64 else ""

    stats = ""
    vol_img = hist_img = ""
    if data:
        d = data
        stats = (
            f'<p style="margin-top:14px">'
            f'<strong>{d["n_green"]}</strong> days kept'
            f' / <strong>{d["n_total"]}</strong>'
            f' ({100*d["n_green"]/d["n_total"]:.1f} %)'
            f' &emsp;|&emsp; Tick : <strong>{d["tick"]:g}</strong>'
            f' &emsp;|&emsp; Valid windows : <strong>{d["n_windows"]}</strong>'
            f'</p>'
        )
        vol_img  = img(d["vol_b64"])
        hist_img = img(d["hist_b64"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Volume viewer</title>
  <style>
    body  {{ font-family: sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 16px; }}
    select, button {{ font-size: 1rem; padding: 6px 10px; }}
    button {{ margin-left: 10px; cursor: pointer; }}
    .tau-row {{ display:flex; align-items:center; gap:10px; margin-top:10px; }}
    input[type=range] {{ width: 200px; }}
    label {{ font-size: 1rem; }}
  </style>
</head>
<body>
  <h2>Volume viewer</h2>
  <form method="get" action="/plot">
    <select name="contract">{contract_opts}</select>
    <div class="tau-row">
      <label>τ = <strong id="tv">{tau}</strong> min</label>
      <input type="range" min="0" max="5" step="1" value="{idx}"
             oninput="
               var v=[1,5,10,15,30,60][+this.value];
               document.getElementById('tv').textContent=v;
               document.getElementById('tau_hidden').value=v;">
      <input type="hidden" id="tau_hidden" name="tau" value="{tau}">
      <button type="submit">Plot</button>
    </div>
  </form>
  {stats}
  {vol_img}
  {hist_img}
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
            params = parse_qs(parsed.query)
            stem   = params.get("contract", [""])[0]
            tau_raw = int(params.get("tau", [5])[0])
            tau     = tau_raw if tau_raw in TAU_VALUES else TAU_VALUES[0]
            match  = next((p for p in self.contracts if p.stem == stem), None)
            if match is None:
                self._send("<p>Contract not found.</p>", 404)
                return
            data = _render(match, tau)
            self._send(_html(self.contracts, selected_stem=stem, tau=tau, data=data))

        else:
            self._send("<p>Not found.</p>", 404)


if __name__ == "__main__":
    port = 8000
    server = HTTPServer(("localhost", port), Handler)
    print(f"Listening on http://localhost:{port}")
    server.serve_forever()
