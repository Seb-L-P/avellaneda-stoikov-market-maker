"""Baseline: a symmetric market maker that ignores inventory entirely.

Deliberately reuses avellaneda_stoikov.optimal_half_spread for the spread
*width* rather than inventing a different one, and only drops the
inventory skew term (quotes are centered on the raw midprice, not the
reservation price). That isolates exactly one thing -- does skewing quotes
against inventory help? -- rather than comparing two strategies that also
happen to quote different widths, which would muddy any P&L/variance
difference with "one strategy just has a wider or narrower spread."
"""

from . import avellaneda_stoikov


def quotes(mid, gamma: float, sigma: float, time_remaining, k: float):
    """Returns (bid, ask), symmetric around the midprice."""
    half = avellaneda_stoikov.optimal_half_spread(gamma, sigma, time_remaining, k)
    return mid - half, mid + half
