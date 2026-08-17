"""Summary statistics from a simulate() result: terminal P&L distribution
and how much inventory risk was actually carried to get there.
"""

import numpy as np


def summarize(result: dict) -> dict:
    terminal_pnl = result["pnl"][:, -1]
    inventory = result["inventory"]
    terminal_inventory = inventory[:, -1]
    max_abs_inventory_per_sim = np.max(np.abs(inventory), axis=1)

    return {
        "mean_pnl": float(terminal_pnl.mean()),
        "std_pnl": float(terminal_pnl.std(ddof=1)),
        "sharpe_like": float(terminal_pnl.mean() / terminal_pnl.std(ddof=1)),
        "mean_terminal_inventory": float(terminal_inventory.mean()),
        "std_terminal_inventory": float(terminal_inventory.std(ddof=1)),
        "mean_abs_inventory_over_time": float(np.mean(np.abs(inventory))),
        "mean_max_abs_inventory": float(max_abs_inventory_per_sim.mean()),
        "clip_binding_frac": result["clip_binding_count"] / result["n_hit_opportunities"],
    }
