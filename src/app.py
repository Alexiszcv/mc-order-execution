"""
Minimal localhost web interface — select a contract, view its volume chart.
Run: python src/app.py
Then open http://localhost:8000
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
from plot_volume import _build_figure, get_tick   # noqa: E402


def _scan_contracts():
    return sorted(
        p for p in DATA.rglob("*.csv") if not p.stem.startswith("AIAgent")
    )


def _render_plot(csv_path: Path) -> tuple:
    """Return (base64 PNG, n_green, n_total, tick)."""
    fig, n_green, n_total = _build_figure(str(csv_path))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    tick = get_tick(str(csv_path))
    return base64.b64encode(buf.getvalue()).decode(), n_green, n_total, tick


def _html(contracts, selected_stem="", img_b64="", n_green=0, n_total=0, tick=None):
    options = "\n".join(
        f'<option value="{p.stem}" {"selected" if p.stem == selected_stem else ""}>'
        f'{p.stem}  ({p.parent.name})</option>'
        for p in contracts
    )
    img_tag = (
        f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%">'
        if img_b64 else ""
    )
    stats = (
        f'<p>'
        f'<strong>{n_green}</strong> day{"s" if n_green != 1 else ""} kept'
        f' out of <strong>{n_total}</strong>'
        f' ({100 * n_green / n_total:.1f} %)&emsp;|&emsp;'
        f'Tick: <strong>{tick:g}</strong>'
        f'</p>'
        if img_b64 else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Volume viewer</title>
  <style>
    body {{ font-family: sans-serif; max-width: 960px; margin: 40px auto; padding: 0 16px; }}
    select, button {{ font-size: 1rem; padding: 6px 10px; }}
    button {{ margin-left: 8px; cursor: pointer; }}
    #chart {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h2>Volume viewer</h2>
  <form method="get" action="/plot">
    <select name="contract">{options}</select>
    <button type="submit">Plot</button>
  </form>
  {stats}
  <div id="chart">{img_tag}</div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    contracts = _scan_contracts()

    def log_message(self, fmt, *args):  # silence default access log
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
            stem = params.get("contract", [""])[0]
            match = next((p for p in self.contracts if p.stem == stem), None)
            if match is None:
                self._send("<p>Contract not found.</p>", 404)
                return
            img_b64, n_green, n_total, tick = _render_plot(match)
            self._send(_html(self.contracts, selected_stem=stem,
                             img_b64=img_b64, n_green=n_green, n_total=n_total, tick=tick))

        else:
            self._send("<p>Not found.</p>", 404)


if __name__ == "__main__":
    port = 8000
    server = HTTPServer(("localhost", port), Handler)
    print(f"Listening on http://localhost:{port}")
    server.serve_forever()
