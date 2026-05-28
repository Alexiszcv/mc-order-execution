"""Result containers for the Monte Carlo layer.

Pure data classes (mirroring the `BacktestResult` style in `order_mgmt.backtest`).
`MCEstimate.from_samples` is the one shared constructor: it turns a sample array into a
point estimate + standard error + confidence interval, either by the CLT (normal SE) or by
empirical percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Side = Literal["buy", "sell"]
CIMethod = Literal["clt", "percentile", "basic"]


@dataclass(frozen=True)
class MCEstimate:
    """A Monte Carlo point estimate with a confidence interval."""

    mean: float
    std_error: float
    ci_low: float
    ci_high: float
    n: int
    method: CIMethod

    @classmethod
    def from_samples(
        cls, samples, *, ci: float = 0.95, method: CIMethod = "clt"
    ) -> MCEstimate:
        """Build an estimate from i.i.d. replicate values (e.g. per-path or per-bootstrap)."""
        arr = np.asarray(samples, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = int(arr.size)
        if n == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0, method)
        mean = float(arr.mean())
        # population->sample std; ddof=1 needs n>=2
        se = float(arr.std(ddof=1) / np.sqrt(n)) if n >= 2 else 0.0
        alpha = 1.0 - ci
        if method == "percentile":
            lo = float(np.quantile(arr, alpha / 2.0))
            hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
        else:  # clt: mean +/- z * se
            from scipy.stats import norm

            z = float(norm.ppf(1.0 - alpha / 2.0))
            lo, hi = mean - z * se, mean + z * se
        return cls(mean=mean, std_error=se, ci_low=lo, ci_high=hi, n=n, method=method)


@dataclass(frozen=True)
class BootstrapResult:
    """Phase A: bootstrapped CIs around the strategy's fill rate and slippage."""

    side: Side
    ell_star: int
    fill_rate: MCEstimate
    avg_slippage_ticks: MCEstimate
    mse_fill_rate: float
    mse_slippage: float
    n_boot: int
    seed: int


@dataclass(frozen=True)
class MCRunResult:
    """Phase B: one forward-simulation run (one range model, one side)."""

    range_model: str  # "empirical" | "fitted" | "gbm" | "bridge"
    side: Side
    ell_star: int
    fill_rate: MCEstimate
    avg_slippage_ticks: MCEstimate
    slippage_samples: list[float]  # full distribution, for histograms
    n_paths: int
    seed: int


@dataclass(frozen=True)
class VRResult:
    """Phase C: a variance-reduction technique applied to an MC estimator."""

    technique: str  # "antithetic" | "control" | "conditional" | "importance"
    estimate: MCEstimate
    baseline_variance: float
    reduced_variance: float
    variance_reduction_ratio: float  # baseline / reduced  (>1 is good)
    n_paths: int
    seed: int
