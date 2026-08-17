"""Avellaneda & Stoikov (2008), "High-frequency trading in a limit order
book" -- the closed-form optimal quoting policy this whole project exists
to test.

The paper solves (via HJB) for the utility-maximizing bid/ask quotes of a
risk-averse market maker who must continuously provide liquidity around a
midprice that follows arithmetic Brownian motion (market.py), with fill
probability decaying exponentially in distance from that midprice:
lambda(delta) = A * exp(-k * delta). Two pieces of closed form fall out:

Reservation price -- the price at which the market maker is indifferent
between holding and not holding one more unit of inventory:

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

This is *not* the midprice; it's the midprice shifted against current
inventory q. Long inventory (q>0) pulls the reservation price down, which
(via the symmetric spread below) makes the ask quote more attractive to
sell into and the bid quote less attractive to buy more into -- the
mechanism that keeps inventory near zero without ever explicitly coding
"if long, sell more."

Optimal spread -- how wide to quote around that reservation price:

    delta_a + delta_b = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

split symmetrically (delta_a = delta_b) around r, as in the paper. The
first term is inventory-risk compensation that shrinks to zero as the
horizon T approaches (less time left to carry risk); the second is a pure
function of order-flow parameters (k, and risk aversion gamma) with no
time dependence -- it's the "compensation for adverse selection" the
market maker demands regardless of how much of the day is left.
"""

import numpy as np


def reservation_price(mid, inventory, gamma: float, sigma: float, time_remaining):
    return mid - inventory * gamma * sigma**2 * time_remaining


def optimal_half_spread(gamma: float, sigma: float, time_remaining, k: float):
    total_spread = gamma * sigma**2 * time_remaining + (2 / gamma) * np.log(1 + gamma / k)
    return total_spread / 2


def quotes(mid, inventory, gamma: float, sigma: float, time_remaining, k: float):
    """Returns (bid, ask)."""
    r = reservation_price(mid, inventory, gamma, sigma, time_remaining)
    half = optimal_half_spread(gamma, sigma, time_remaining, k)
    return r - half, r + half
