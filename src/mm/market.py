"""The reference ("true") midprice process the market maker quotes around.

Avellaneda-Stoikov (2008) uses arithmetic Brownian motion for the midprice
-- dS_t = sigma dW_t, no drift -- rather than geometric Brownian motion.
That's a deliberate modeling choice in the original paper, not corner-
cutting: the reservation-price and optimal-spread formulas in
avellaneda_stoikov.py are derived in closed form specifically for constant
(additive, not multiplicative) volatility, and changing this to GBM would
mean re-deriving the HJB solution, not just swapping the simulator.

Fully vectorized across simulations, same cumsum-of-increments pattern used
for path simulation in the options-pricing project's LSM module -- no
Python-level loop over time at all here, since (unlike the market-making
simulator itself) nothing about the midprice depends on anything the
simulator decides.
"""

import numpy as np


def simulate_midprice(n_sims: int, n_steps: int, dt: float, s0: float, sigma: float, seed: int) -> np.ndarray:
    """Returns an (n_sims, n_steps+1) array of midprice paths."""
    rng = np.random.default_rng(seed)
    increments = sigma * np.sqrt(dt) * rng.standard_normal((n_sims, n_steps))
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 + np.cumsum(increments, axis=1)
    return paths
