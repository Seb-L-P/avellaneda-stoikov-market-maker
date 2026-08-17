# Market Making Simulator (Avellaneda-Stoikov)

An implementation of Avellaneda & Stoikov's (2008) optimal market-making model — a risk-averse market maker continuously quoting bid/ask prices around a random-walk midprice, skewing quotes against current inventory to keep risk under control — benchmarked against a spread-matched naive baseline across tens of thousands of simulated trading days.

Third in a series alongside a Monte Carlo blackjack/Kelly-sizing project and an options pricing & Greeks engine. Where those were "does my simulation match a known number," this one is closer to "does a closed-form optimal *policy* actually outperform a naive one, and by how much" — a genuine stochastic control result, not just a pricing formula.

## CV bullet

> **Market Making Simulator (Avellaneda-Stoikov) — Python · [GitHub]**
> - Implemented the Avellaneda-Stoikov (2008) optimal market-making model (closed-form inventory-skewed quoting under Poisson order arrivals) with a fully vectorized order-arrival simulation engine, running 50,000 parallel simulated trading days per experiment
> - Built a spread-matched naive baseline (identical quote width, no inventory skew) to isolate the value of inventory-aware quoting specifically, using common random numbers so both strategies face identical order flow
> - Result: inventory-skewed quoting cut P&L standard deviation by 51% and terminal inventory standard deviation by 65% for comparable mean P&L (~5% apart), nearly doubling the Sharpe-like ratio (9.9 vs. 5.1); a risk-aversion sweep revealed a non-monotonic risk/return tradeoff with an interior-optimal γ — more risk aversion isn't always better, even for risk-adjusted return

## Results

Parameters throughout are Avellaneda & Stoikov's own 2008 numerical example: S0=100, σ=2, T=1 (200 steps of dt=0.005), γ=0.1, k=1.5, A=140. Reproduce with `python scripts/run_experiment.py`.

### Avellaneda-Stoikov vs. a spread-matched naive baseline

| | Avellaneda-Stoikov | Naive symmetric |
|---|---:|---:|
| Mean P&L | 64.86 | 67.92 |
| P&L std dev | 6.53 | 13.40 |
| Sharpe-like (mean/std) | 9.93 | 5.07 |
| Terminal inventory std dev | 2.92 | 8.42 |
| Mean \|inventory\| over the day | 0.98 | 4.37 |

The naive baseline isn't a strawman — it uses the *exact same* optimal-spread-width formula, just centered on the raw midprice instead of the inventory-adjusted reservation price. So this table isolates one thing precisely: **what does skewing your quotes against inventory buy you**, holding spread width and order flow fixed. Answer: essentially the same average revenue, for roughly half the P&L volatility.

![Sample inventory paths](results/sample_inventory_paths.png)

This is the clearest single result. Left panel: AS inventory oscillates in a tight band around zero all day, visibly pulled back whenever it drifts. Right panel: naive inventory does an unconstrained random walk and several paths wander to ±20-28 units by end of day — real, uncompensated directional exposure that has nothing to do with market making and everything to do with the midprice happening to drift.

![P&L distribution](results/pnl_distribution.png)

### Risk-aversion sweep — and a non-obvious finding

| γ | Mean P&L | P&L std | Sharpe-like | Mean \|inventory\| |
|---:|---:|---:|---:|---:|
| 0.01 | 68.35 | 9.00 | 7.60 | 2.58 |
| 0.02 | 67.99 | 7.91 | 8.60 | 1.98 |
| 0.05 | 66.84 | 6.96 | 9.61 | 1.35 |
| **0.10** | **64.86** | **6.53** | **9.93** | **0.98** |
| 0.20 | 60.80 | 6.26 | 9.71 | 0.69 |
| 0.30 | 56.77 | 6.13 | 9.26 | 0.55 |
| 0.50 | 48.51 | 5.87 | 8.27 | 0.38 |
| 0.80 | 37.25 | 5.24 | 7.11 | 0.26 |
| 1.20 | 26.86 | 4.43 | 6.07 | 0.18 |

![Efficient frontier](results/efficient_frontier.png)

Mean P&L and inventory both decrease monotonically as γ increases — no surprise, more risk aversion means quoting wider and holding less inventory, at the cost of less spread captured. What *isn't* obvious ahead of time: the **Sharpe-like ratio peaks around γ≈0.1 and declines on both sides of it.** Too little risk aversion leaves real, uncompensated inventory risk on the table; too much sacrifices so much mean P&L chasing ever-smaller inventory that risk-adjusted performance gets worse too, not better. There's an interior optimum, not a monotonic "more caution is always better" relationship — genuinely the same shape as an efficient frontier in portfolio theory, which makes sense: this *is* a risk-aversion-parameterized allocation problem, just for inventory instead of asset weights.

![Spread vs volatility](results/spread_vs_sigma.png)

Pure formula sanity check, no simulation involved: quoted spread should widen as volatility rises (more inventory risk per unit time → charge more to bear it), and it does, roughly quadratically — consistent with the `γσ²(T−t)` term in the optimal-spread formula.

## Methodology

**Midprice (`market.py`).** Arithmetic Brownian motion, `dS_t = σ dW_t`, no drift — the model in the AS paper, not a simplification of it. The reservation-price and optimal-spread formulas below are derived (via HJB) specifically for *additive*, not multiplicative, volatility; swapping in GBM would mean re-deriving the closed form, not just changing the simulator. Fully vectorized: one `cumsum` of increments produces all simulated trading days' midprice paths at once, no time loop at all (the same pattern used for GBM path simulation in the options-pricing project's LSM module).

**The policy (`avellaneda_stoikov.py`).** Two pieces of closed form from the paper:
- **Reservation price** `r = s − qγσ²(T−t)`: the price at which the market maker is indifferent to holding one more unit of inventory. Not the midprice — shifted against current inventory `q`. Long inventory pulls it down; short inventory pushes it up.
- **Optimal spread** `δ_a+δ_b = γσ²(T−t) + (2/γ)ln(1+γ/k)`, split symmetrically around the reservation price. The first term is inventory-risk compensation that shrinks to zero as the trading horizon closes (less time left to carry risk); the second is a pure order-flow/risk-aversion term with no time dependence.

Quoting the reservation price instead of the midprice, symmetrically, is what makes the *ask* more attractive to sell into and the *bid* less attractive to buy more into whenever inventory is long — the mechanism that keeps inventory near zero without ever coding an explicit "if long, sell more" rule. It falls out of the price alone.

**Order arrivals and the simulation loop (`simulator.py`).** A resting quote a distance δ below/above the midprice gets hit with probability ≈`λ(δ)·dt` per step, where `λ(δ)=A·e^{−kδ}` — the standard first-order discretization of two independent Poisson processes (bid-side and ask-side arrivals), each drawn as an *independent* Bernoulli trial per step (not mutually exclusive — both a buy and a sell can arrive in the same step, which is realistic and just nets to a wash on inventory while still banking the spread). Same "loop over time, vectorize across simulations" pattern used for the binomial tree and LSM regression in the options project and the block-bootstrap in the blackjack project: the only Python-level loop is over the 200 time steps, and every step updates all 50,000 simulated trading days simultaneously via NumPy.

**Common random numbers, twice over.** `simulate()` reseeds its own RNG from a fixed `seed`, so running it once with the inventory skew on and once off reuses *identical* midprice paths and order-arrival draws in both runs — any resulting difference in P&L/inventory is attributable to the quoting policy, not to one run getting luckier order flow. The same trick isolates the risk-aversion sweep: every γ in the sweep sees the same midprice paths and arrival draws, so the sweep traces out the effect of γ alone.

## Validating against the paper

Rather than trust the implementation on formula inspection alone, the numbers were checked against the *qualitative* result the original paper reports for this exact parameter set: comparable mean P&L between the AS and symmetric strategies, meaningfully lower P&L and inventory variance for AS, and inventory that visibly mean-reverts rather than drifts. All of that reproduced directly (see the results above) — a stronger check than matching one absolute number, for the same reason cross-checking three independent rule-variant effects was stronger evidence than one house-edge number in the blackjack project.

The probability-clipping fallback (`p_bid`/`p_ask` clipped to `[0,1]`, needed because `λ·dt` can technically exceed 1 for extreme inventory/parameter combinations) binds on **0.008% of bid/ask opportunities** — confirmation that the chosen `dt=0.005` is fine-grained enough for the Poisson approximation to hold almost everywhere, not a crutch papering over an unstable parameter regime.

## Assumptions & limitations

- **Reduced-form order arrivals, not a reconstructed limit order book.** This is deliberate, not a shortcut — it's the model in the original paper. A full LOB simulator (resting orders from other participants, price-time priority, realistic order-flow dynamics) would be a substantially larger, different project; see Extensions.
- **Symmetric spread split** (`δ_a = δ_b`) around the reservation price, as in the paper's own presentation, rather than the fully general asymmetric solution.
- **Fixed γ, k, A, σ for the whole trading day** — no regime changes, no intraday seasonality in order flow (real markets are busier at the open/close), no adverse-selection-driven widening around news events.
- **Inventory is unbounded** — no hard position limits are enforced (the AS policy discourages large inventory but doesn't forbid it), and no capital/margin constraint is modeled.
- **Mark-to-market P&L** values terminal inventory at the final simulated midprice; it doesn't model the cost of actually unwinding a residual position (crossing the spread to flatten at day's end).

## Project structure

```
src/mm/
  market.py               vectorized midprice random walk (arithmetic BM)
  avellaneda_stoikov.py   reservation price + optimal spread closed form
  naive.py                 spread-matched symmetric baseline
  simulator.py             vectorized order-arrival simulation, inventory/cash/P&L
  metrics.py               summary stats: P&L, inventory risk, Sharpe-like ratio
  plotting.py               report plots
scripts/
  run_experiment.py        AS vs. naive, risk-aversion sweep, volatility sanity check
tests/                      martingale/variance checks, AS formula monotonicity,
                            P&L accounting consistency, AS-vs-naive risk reduction
results/                    generated plots, CSVs, and summary.json from the run above
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -v
python scripts/run_experiment.py
```

## Possible extensions

- A real limit order book with resting orders from other simulated participants, rather than the reduced-form Poisson arrival model
- Position limits / hard inventory caps, and the resulting quote behavior at the boundary
- Regime-switching or time-of-day-varying order flow (A, k) instead of constants
- The fully general asymmetric bid/ask solution instead of the symmetric-split approximation
- Adverse selection: let the midprice's next move be predictable from recent order flow, and see how much the naive strategy's performance degrades relative to AS when being picked off is a real risk
