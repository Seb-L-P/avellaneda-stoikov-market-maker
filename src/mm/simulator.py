"""Order-arrival simulation against a quoting policy.

At each time step, a resting bid a distance delta_bid below the midprice
gets hit by an incoming sell (the market maker buys) with probability
~lambda(delta_bid)*dt, and symmetrically for the ask -- the standard
first-order discretization of two independent Poisson processes with
intensity lambda(delta) = A*exp(-k*delta): P(>=1 arrival in dt) ~ lambda*dt
for small dt, P(>=2 arrivals in dt) is O(dt^2) and dropped. Bid and ask
fills are drawn as *independent* Bernoulli trials, not mutually exclusive
-- both can fire in the same step (a buy and a sell both arriving), which
is realistic (not a bug) and simply nets to a wash on inventory while
still capturing the spread as P&L, exactly as it would if both trades hit
in quick succession in reality.

Like the binomial tree and the LSM path regression in the options-pricing
project, and the block-bootstrap in the blackjack project, the only
Python-level loop here is over time (n_steps, typically ~200) -- every
step is vectorized across all n_sims simulated trading days at once.

Common random numbers: `simulate()` reseeds its own RNG from `seed` and
draws in the same fixed order every call, so running it once with
use_inventory_skew=True and once with False on the *same* seed reuses
identical order-arrival draws in both runs. Any difference between the two
resulting P&L/inventory distributions is then attributable to the quoting
policy itself, not to one run happening to get luckier order flow.
"""

import numpy as np

from . import avellaneda_stoikov, naive


def simulate(
    midprice_paths: np.ndarray, dt: float, T: float, gamma: float, sigma: float,
    k: float, A: float, use_inventory_skew: bool, seed: int,
) -> dict:
    n_sims, n_plus1 = midprice_paths.shape
    n_steps = n_plus1 - 1
    rng = np.random.default_rng(seed)

    inventory = np.zeros(n_sims)
    cash = np.zeros(n_sims)
    inventory_path = np.zeros((n_sims, n_plus1))
    cash_path = np.zeros((n_sims, n_plus1))
    clip_binding_count = 0

    for step in range(n_steps):
        time_remaining = T - step * dt
        mid = midprice_paths[:, step]

        if use_inventory_skew:
            bid, ask = avellaneda_stoikov.quotes(mid, inventory, gamma, sigma, time_remaining, k)
        else:
            bid, ask = naive.quotes(mid, gamma, sigma, time_remaining, k)

        delta_bid = mid - bid
        delta_ask = ask - mid
        lam_bid = A * np.exp(-k * delta_bid)
        lam_ask = A * np.exp(-k * delta_ask)
        p_bid = lam_bid * dt
        p_ask = lam_ask * dt
        clip_binding_count += int(np.sum((p_bid > 1) | (p_ask > 1)))
        p_bid = np.clip(p_bid, 0.0, 1.0)
        p_ask = np.clip(p_ask, 0.0, 1.0)

        bid_hit = rng.random(n_sims) < p_bid
        ask_hit = rng.random(n_sims) < p_ask

        inventory = inventory + bid_hit - ask_hit
        cash = cash - bid * bid_hit + ask * ask_hit

        inventory_path[:, step + 1] = inventory
        cash_path[:, step + 1] = cash

    pnl_path = cash_path + inventory_path * midprice_paths

    return {
        "midprice": midprice_paths,
        "inventory": inventory_path,
        "cash": cash_path,
        "pnl": pnl_path,
        "clip_binding_count": clip_binding_count,
        "n_hit_opportunities": n_sims * n_steps * 2,
    }
