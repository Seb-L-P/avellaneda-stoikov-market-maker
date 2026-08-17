from mm.avellaneda_stoikov import optimal_half_spread, quotes, reservation_price

BASE = dict(gamma=0.1, sigma=2.0, time_remaining=1.0, k=1.5)


def test_reservation_price_equals_mid_at_zero_inventory():
    r = reservation_price(mid=100.0, inventory=0, gamma=0.1, sigma=2.0, time_remaining=1.0)
    assert r == 100.0


def test_long_inventory_pulls_reservation_price_below_mid():
    r = reservation_price(mid=100.0, inventory=5, gamma=0.1, sigma=2.0, time_remaining=1.0)
    assert r < 100.0


def test_short_inventory_pulls_reservation_price_above_mid():
    r = reservation_price(mid=100.0, inventory=-5, gamma=0.1, sigma=2.0, time_remaining=1.0)
    assert r > 100.0


def test_spread_widens_with_volatility():
    narrow = optimal_half_spread(**{**BASE, "sigma": 1.0})
    wide = optimal_half_spread(**{**BASE, "sigma": 3.0})
    assert wide > narrow


def test_spread_widens_with_risk_aversion():
    narrow = optimal_half_spread(**{**BASE, "gamma": 0.05})
    wide = optimal_half_spread(**{**BASE, "gamma": 0.3})
    assert wide > narrow


def test_spread_narrows_as_horizon_approaches():
    # time_remaining -> 0 means less time left to carry inventory risk, so
    # the risk-compensation term of the spread shrinks toward the pure
    # order-flow term.
    start_of_day = optimal_half_spread(**{**BASE, "time_remaining": 1.0})
    near_close = optimal_half_spread(**{**BASE, "time_remaining": 0.001})
    assert near_close < start_of_day


def test_spread_narrows_with_deeper_book_k():
    thin_book = optimal_half_spread(**{**BASE, "k": 0.5})
    deep_book = optimal_half_spread(**{**BASE, "k": 5.0})
    assert deep_book < thin_book


def test_quotes_bracket_reservation_price_symmetrically():
    bid, ask = quotes(mid=100.0, inventory=3, **BASE)
    r = reservation_price(mid=100.0, inventory=3, gamma=BASE["gamma"], sigma=BASE["sigma"], time_remaining=BASE["time_remaining"])
    assert abs((r - bid) - (ask - r)) < 1e-12
    assert bid < r < ask
