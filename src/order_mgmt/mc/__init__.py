"""Monte Carlo execution layer.

Forward-simulates the regime-conditioned limit-order strategy and quantifies its
fill-rate / slippage with confidence intervals. Three ways to model the per-regime range
(empirical resampling, fitted parametric distribution, Brownian-motion path) are compared
against the historical backtest. See plan: stream/f-mc.
"""
