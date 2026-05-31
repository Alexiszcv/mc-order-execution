"""Per-market tick-size table + heuristic inference.

`resolve_tick` prefers the table value and falls back to `infer_tick` (smallest observed
price gap) only for markets absent from `TICK_TABLE`.

Sources: the assignment PDF (§3.2) fixes only ES = 0.25 and delegates the rest to each
contract's exchange spec. `TICK_TABLE` holds the tick **in the provided CSVs' own quoting
units** — because every downstream consumer (ranges, ePDFs, backtest, agent eval) needs
ε in the data's units for ℓ = ΔPrice/ε to be a count of spreads. Each value is the exchange
spec tick adjusted for the CSV's quoting scale, and cross-checked against the data via
`infer_tick` (smallest observed price gap over the full history):

| Contract | Exchange spec tick | CSV quoting scale | Data-unit tick |
|----------|--------------------|-------------------|----------------|
| GC (COMEX Gold)        | 0.10 USD/oz       | as-is        | 0.10  |
| ES / NQ (CME E-mini)   | 0.25 index pts    | as-is        | 0.25  |
| RX (Eurex Bund)        | 0.01 price pts    | as-is        | 0.01  |
| BP (GBP/USD)           | 0.0001 USD/GBP    | ×100 (1.37→137)  | 0.01  |
| JY (JPY/USD)           | 0.0000005 USD/JPY | ×10000           | 0.005 |
| HO (NYMEX ULSD)        | 0.0001 USD/gal    | cents/gal (×100) | 0.01  |
| VG (EuroStoxx 50)      | 1.0 index pt      | 0.5 prints in CSV| 0.5   |

Earlier versions stored the *raw exchange* tick for the scaled markets (BP/JY/HO/VG), which
silently inflated ℓ by the scale factor (e.g. GBP ℓ ×100). The values below are in data
units so `resolve_tick` is safe to use directly for all markets.
"""

from __future__ import annotations

from math import floor, log10

import numpy as np

# Values are in the provided CSVs' quoting units (see module docstring), verified via
# infer_tick on the full price history of each market.
TICK_TABLE: dict[str, float] = {
    "GC": 0.10,    # COMEX Gold, as-is
    "NQ": 0.25,    # CME E-mini Nasdaq-100, as-is
    "ES": 0.25,    # CME E-mini S&P 500, as-is
    "RX": 0.01,    # Eurex Bund, as-is
    "BP": 0.01,    # GBP/USD: 0.0001 USD/GBP × ×100 CSV scale
    "JY": 0.005,   # JPY/USD: 0.0000005 USD/JPY × ×10000 CSV scale
    "HO": 0.01,    # NYMEX ULSD: 0.0001 USD/gal in cents/gal
    "VG": 0.5,     # EuroStoxx 50: CSV prints on a 0.5 grid
}


def _alpha_prefix(stem: str) -> str:
    out = []
    for ch in stem:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out)


def lookup_tick(contract_stem: str) -> float | None:
    """Return the spec tick size for a contract stem (e.g. 'GCM24') or None if unknown.

    Matches by alphabetic prefix; e.g. 'GCM24' → 'GC' → 0.10.
    """
    prefix = _alpha_prefix(contract_stem)
    if not prefix:
        return None
    for k in sorted(TICK_TABLE.keys(), key=len, reverse=True):
        if prefix.startswith(k):
            return TICK_TABLE[k]
    return None


def resolve_tick(contract_stem: str, fallback: float) -> float:
    """Spec tick if known for this contract, else `fallback` (typically the inferred value)."""
    spec = lookup_tick(contract_stem)
    return spec if spec is not None else fallback


def infer_tick(prices) -> float:
    """Infer the tick size as the smallest positive gap between observed price levels.

    `prices` is any array-like of price observations (typically every OHLC value of a
    contract). This is the heuristic fallback for markets absent from `TICK_TABLE`;
    prefer `resolve_tick`, which only falls back to this when the spec value is unknown.
    """
    levels = np.unique(np.round(np.asarray(prices, dtype=float), 8))
    diffs = np.diff(levels)
    positive = diffs[diffs > 1e-9]
    if positive.size == 0:
        raise ValueError("cannot infer tick: fewer than two distinct prices")
    raw = float(positive.min())
    # round to the gap's own magnitude (+7 digits) to shed float noise, e.g. 0.0999999 -> 0.1
    mag = floor(log10(raw))
    return round(raw, -mag + 7)
