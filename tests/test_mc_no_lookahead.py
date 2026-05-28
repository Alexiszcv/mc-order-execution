"""MC layer no-lookahead boundary: pure function of inputs; bridge rejected; order-free."""

from __future__ import annotations

from collections import Counter

from order_mgmt.mc.simulator import run_mc_execution


def _epdf() -> Counter:
    return Counter({0: 30, 1: 25, 2: 20, 3: 15, 4: 10})


def test_deterministic_in_seed() -> None:
    kw = dict(
        side="sell", fill_rate_target=0.5, n_paths=5000,
        range_model="gbm", tau=5, sigma_bar=3.0, seed=7,
    )
    a = run_mc_execution(_epdf(), **kw)
    b = run_mc_execution(_epdf(), **kw)
    assert a.fill_rate.mean == b.fill_rate.mean
    assert a.slippage_samples == b.slippage_samples


def test_bridge_rejected_in_decision_path() -> None:
    import pytest

    with pytest.raises(ValueError, match="look-ahead"):
        run_mc_execution(
            _epdf(), side="sell", fill_rate_target=0.5, n_paths=10,
            range_model="bridge", tau=5, sigma_bar=3.0, seed=0,
        )


def test_input_order_invariant() -> None:
    """The ePDF is a multiset; the order it was assembled in must not change the output."""
    e1 = Counter()
    for v in [0, 0, 1, 2, 2, 2, 4]:
        e1[v] += 1
    e2 = Counter()
    for v in [4, 2, 0, 2, 1, 2, 0]:  # same multiset, shuffled insertion order
        e2[v] += 1
    kw = dict(
        side="buy", fill_rate_target=0.6, n_paths=4000,
        range_model="empirical", tau=5, sigma_bar=2.5, seed=3,
    )
    r1 = run_mc_execution(e1, **kw)
    r2 = run_mc_execution(e2, **kw)
    assert r1.slippage_samples == r2.slippage_samples
