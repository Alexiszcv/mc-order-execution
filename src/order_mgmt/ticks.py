"""Per-market tick-size table.

The spec gives explicit tick sizes per contract; the table here is the canonical source.
The team's heuristic (`plot_volume.py:get_tick`) is kept as a fallback for unknown
markets via `resolve_tick`.

Sources:
- COMEX Gold (GC) = 0.10 USD/oz
- CME E-mini equity index: ES (S&P) and NQ (Nasdaq-100) = 0.25
- CME FX: BP (GBP/USD) = 0.0001 USD/GBP, JY (JPY/USD) = 0.000001 USD/JPY
- Eurex Bund (RX) = 0.01 (price points), EuroStoxx 50 (VG) = 1.0 (index points)
- NYMEX heating oil (HO) = 0.0001 USD/gal
"""

from __future__ import annotations


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
