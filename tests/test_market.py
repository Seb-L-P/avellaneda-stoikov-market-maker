import numpy as np

from mm.market import simulate_midprice


def test_midprice_is_a_martingale():
    paths = simulate_midprice(n_sims=200_000, n_steps=200, dt=0.005, s0=100.0, sigma=2.0, seed=0)
    terminal = paths[:, -1]
    stderr = terminal.std(ddof=1) / np.sqrt(len(terminal))
    assert abs(terminal.mean() - 100.0) < 4 * stderr


def test_midprice_terminal_variance_matches_sigma_squared_T():
    T = 1.0
    sigma = 2.0
    paths = simulate_midprice(n_sims=200_000, n_steps=200, dt=T / 200, s0=100.0, sigma=sigma, seed=1)
    terminal = paths[:, -1]
    expected_var = sigma**2 * T
    # sample variance of a sample variance has its own (larger) sampling error;
    # 5% relative tolerance at 200k paths is comfortably outside that noise band.
    assert abs(terminal.var(ddof=1) - expected_var) / expected_var < 0.05
