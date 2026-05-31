"""AIAgent execution-value pipeline.

Evaluates how much the regime-conditioned limit-order execution improves the
"AI agent's" trading vs naive market-on-decision execution, using only OHLC data.

Submodules:
  - loader    : decode AIAgent_*.csv (Excel serial day) -> tidy trade decisions
  - metrics   : per-decision implementation-shortfall + value-add vs baselines
  - synthetic : seeded synthetic AIAgent series (zero-shot / new-data stress test)
  - slicing   : order-slicing fill schemes (single / time_slice / blend / cutoff)
  - dynamic   : sequential execution (adaptive shrinking-ℓ* rule + DP optimal stopping)

No order-book data exists, so these metrics measure implementation shortfall
(execution price vs arrival price), not market-impact slippage. See metrics.py.
"""

from __future__ import annotations

__all__ = ["dynamic", "loader", "metrics", "slicing", "synthetic"]
