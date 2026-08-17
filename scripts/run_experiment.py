#!/usr/bin/env python
"""End-to-end experiment: simulate a market maker quoting under the
Avellaneda-Stoikov policy vs. a spread-matched symmetric baseline, sweep
risk aversion to trace a risk/return frontier, and sanity-check the
closed-form spread formula's sensitivity to volatility. Writes plots,
CSVs, and a summary.json to --outdir.

Default parameters (T=1, dt=0.005, sigma=2, gamma=0.1, k=1.5, A=140) are
the ones used in Avellaneda & Stoikov's own 2008 paper.

    python scripts/run_experiment.py
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mm.avellaneda_stoikov import optimal_half_spread
from mm.market import simulate_midprice
from mm.metrics import summarize
from mm.plotting import (
    plot_efficient_frontier,
    plot_pnl_distribution,
    plot_sample_inventory_paths,
    plot_spread_vs_sigma,
)
from mm.simulator import simulate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sims", type=int, default=50_000)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--s0", type=float, default=100.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--k", type=float, default=1.5)
    p.add_argument("--A", type=float, default=140.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dt = args.T / args.n_steps

    print(f"Simulating {args.n_sims:,} trading days x {args.n_steps} steps...")
    t0 = time.time()
    midprice_paths = simulate_midprice(args.n_sims, args.n_steps, dt, args.s0, args.sigma, seed=args.seed)

    common = dict(T=args.T, dt=dt, gamma=args.gamma, sigma=args.sigma, k=args.k, A=args.A, seed=args.seed)
    as_result = simulate(midprice_paths, use_inventory_skew=True, **common)
    naive_result = simulate(midprice_paths, use_inventory_skew=False, **common)
    print(f"  done in {time.time() - t0:.1f}s")

    as_stats = summarize(as_result)
    naive_stats = summarize(naive_result)
    comparison = pd.DataFrame({"avellaneda_stoikov": as_stats, "naive_symmetric": naive_stats})
    comparison.to_csv(os.path.join(args.outdir, "as_vs_naive.csv"))
    print("\n--- Avellaneda-Stoikov vs. naive symmetric (same spread width, same order flow) ---")
    print(comparison.to_string())

    plot_sample_inventory_paths(as_result, naive_result, dt, os.path.join(args.outdir, "sample_inventory_paths.png"))
    plot_pnl_distribution(
        as_result["pnl"][:, -1], naive_result["pnl"][:, -1], os.path.join(args.outdir, "pnl_distribution.png")
    )

    # --- Risk-aversion sweep: efficient frontier ---
    print("\nSweeping risk aversion (gamma)...")
    gamma_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    frontier_rows = []
    for g in gamma_grid:
        r = simulate(midprice_paths, T=args.T, dt=dt, gamma=g, sigma=args.sigma, k=args.k, A=args.A,
                      use_inventory_skew=True, seed=args.seed)
        s = summarize(r)
        frontier_rows.append({"gamma": g, **s})
    frontier_df = pd.DataFrame(frontier_rows)
    frontier_df.to_csv(os.path.join(args.outdir, "gamma_sweep.csv"), index=False)
    plot_efficient_frontier(
        frontier_df["mean_pnl"], frontier_df["std_pnl"], frontier_df["gamma"],
        os.path.join(args.outdir, "efficient_frontier.png"),
    )
    print(frontier_df[["gamma", "mean_pnl", "std_pnl", "sharpe_like", "mean_abs_inventory_over_time"]].to_string(index=False))

    # --- Adverse selection sweep: what informed flow does to both policies ---
    print("\nSweeping adverse selection (permanent impact per fill, price units)...")
    xi_grid = [0.0, 0.02, 0.05, 0.1, 0.2]
    adverse_rows = []
    for xi in xi_grid:
        for label, skew in (("avellaneda_stoikov", True), ("naive_symmetric", False)):
            r = simulate(midprice_paths, use_inventory_skew=skew, adverse_selection=xi, **common)
            s = summarize(r)
            adverse_rows.append({"adverse_selection": xi, "policy": label,
                                 "mean_pnl": s["mean_pnl"], "std_pnl": s["std_pnl"],
                                 "sharpe_like": s["sharpe_like"]})
    adverse_df = pd.DataFrame(adverse_rows)
    adverse_df.to_csv(os.path.join(args.outdir, "adverse_selection_sweep.csv"), index=False)
    print(adverse_df.to_string(index=False))

    # --- Volatility sensitivity (pure formula check, no simulation needed) ---
    sigma_grid = np.linspace(0.5, 4.0, 15)
    spreads = [2 * optimal_half_spread(gamma=args.gamma, sigma=s, time_remaining=args.T, k=args.k) for s in sigma_grid]
    plot_spread_vs_sigma(sigma_grid, spreads, os.path.join(args.outdir, "spread_vs_sigma.png"))

    summary = {
        "params": {"T": args.T, "dt": dt, "n_steps": args.n_steps, "n_sims": args.n_sims,
                    "s0": args.s0, "sigma": args.sigma, "gamma": args.gamma, "k": args.k, "A": args.A},
        "avellaneda_stoikov": as_stats,
        "naive_symmetric": naive_stats,
        "pnl_std_reduction_pct": 100 * (1 - as_stats["std_pnl"] / naive_stats["std_pnl"]),
        "inventory_std_reduction_pct": 100 * (1 - as_stats["std_terminal_inventory"] / naive_stats["std_terminal_inventory"]),
        "sharpe_like_improvement_pct": 100 * (as_stats["sharpe_like"] / naive_stats["sharpe_like"] - 1),
        "gamma_sweep": frontier_rows,
        "adverse_selection_sweep": adverse_rows,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Headline results ---")
    print(f"P&L std reduction (AS vs. naive):        {summary['pnl_std_reduction_pct']:.1f}%")
    print(f"Terminal inventory std reduction:        {summary['inventory_std_reduction_pct']:.1f}%")
    print(f"Sharpe-like ratio improvement:           {summary['sharpe_like_improvement_pct']:.1f}%")
    print(f"\nWrote plots/CSVs/summary to {args.outdir}/")


if __name__ == "__main__":
    main()
