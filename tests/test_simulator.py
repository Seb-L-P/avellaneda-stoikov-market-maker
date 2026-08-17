import numpy as np

from mm.market import simulate_midprice
from mm.simulator import simulate

PARAMS = dict(T=1.0, dt=0.005, gamma=0.1, sigma=2.0, k=1.5, A=140.0)


def _run(n_sims, use_inventory_skew, seed):
    midprice_paths = simulate_midprice(n_sims, n_steps=200, dt=PARAMS["dt"], s0=100.0, sigma=PARAMS["sigma"], seed=seed)
    return simulate(midprice_paths, use_inventory_skew=use_inventory_skew, seed=seed, **PARAMS)


def test_pnl_accounting_is_internally_consistent():
    result = _run(n_sims=500, use_inventory_skew=True, seed=0)
    reconstructed = result["cash"] + result["inventory"] * result["midprice"]
    assert np.allclose(reconstructed, result["pnl"])


def test_avellaneda_stoikov_reduces_inventory_risk_vs_naive():
    n_sims = 20_000
    as_result = _run(n_sims, use_inventory_skew=True, seed=42)
    naive_result = _run(n_sims, use_inventory_skew=False, seed=42)  # same seed: common random numbers

    as_std = as_result["inventory"][:, -1].std(ddof=1)
    naive_std = naive_result["inventory"][:, -1].std(ddof=1)
    assert as_std < naive_std


def test_avellaneda_stoikov_and_naive_have_comparable_mean_pnl():
    # The comparison is only meaningful if it isn't secretly comparing a
    # wider spread against a narrower one -- both use the same spread
    # *width* formula, so mean P&L (spread capture) should be in the same
    # ballpark even though variance differs a lot.
    n_sims = 20_000
    as_result = _run(n_sims, use_inventory_skew=True, seed=7)
    naive_result = _run(n_sims, use_inventory_skew=False, seed=7)

    as_mean = as_result["pnl"][:, -1].mean()
    naive_mean = naive_result["pnl"][:, -1].mean()
    assert abs(as_mean - naive_mean) < 0.25 * max(abs(as_mean), abs(naive_mean))


def test_higher_risk_aversion_reduces_inventory_variance():
    n_sims = 10_000
    midprice_paths = simulate_midprice(n_sims, n_steps=200, dt=PARAMS["dt"], s0=100.0, sigma=PARAMS["sigma"], seed=3)
    low_gamma = simulate(midprice_paths, T=1.0, dt=0.005, gamma=0.02, sigma=2.0, k=1.5, A=140.0, use_inventory_skew=True, seed=3)
    high_gamma = simulate(midprice_paths, T=1.0, dt=0.005, gamma=0.5, sigma=2.0, k=1.5, A=140.0, use_inventory_skew=True, seed=3)

    low_gamma_std = low_gamma["inventory"][:, -1].std(ddof=1)
    high_gamma_std = high_gamma["inventory"][:, -1].std(ddof=1)
    assert high_gamma_std < low_gamma_std
