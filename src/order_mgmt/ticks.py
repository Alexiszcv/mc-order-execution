"""Per-market tick-size table + heuristic inference.

`resolve_tick` prefers the table value and falls back to `infer_tick` (smallest observed
price gap) only for markets absent from `TICK_TABLE`.

Sources: the assignment PDF (§3.2) fixes only ES = 0.25 and delegates the rest to each
contract's exchange spec. The remaining values are from the listed exchange contract
specs, cross-checked empirically against the provided 1-min data via `infer_tick`:
- COMEX Gold (GC) = 0.10 USD/oz
- CME E-mini equity index: ES (S&P) and NQ (Nasdaq-100) = 0.25
- CME FX: BP (GBP/USD) = 0.0001 USD/GBP, JY (JPY/USD) = 0.000001 USD/JPY
- Eurex Bund (RX) = 0.01 (price points), EuroStoxx 50 (VG) = 1.0 (index points)
- NYMEX heating oil (HO) = 0.0001 USD/gal

Caveat — quoting scale: the table is in *raw exchange units*, but the provided CSVs quote
some markets at a scaled representation, so the table value and the data granularity differ
by a scale factor: GBP is quoted x100 (1.3704 -> 137.04, tick 0.0001 -> 0.01), JPY x10000,
HO in cents, and VG shows 0.5 prints. For range/spread math (l = dPrice/eps) eps must be in
the data's own units, so use `infer_tick` on the actual data; treat `TICK_TABLE` as a
documentation/sanity reference, not an override. Wiring the raw table value into the spread
counts would silently inflate l by the scale factor for GBP/JPY/HO/VG.
"""

from __future__ import annotations

from math import floor, log10

import numpy as np

TICK_TABLE: dict[str, float] = {
    "GC": 0.10,
    "NQ": 0.25,
    "ES": 0.25,
    "BP": 0.0001,
    "JY": 0.000001,
    "RX": 0.01,
    "VG": 1.0,
    "HO": 0.0001,
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
