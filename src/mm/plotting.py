"""Report plots. Agg backend so this renders headless."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sample_inventory_paths(result_as: dict, result_naive: dict, dt: float, path: str, n_show: int = 30) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    n_steps = result_as["inventory"].shape[1]
    t = np.arange(n_steps) * dt
    idx = np.random.default_rng(0).choice(result_as["inventory"].shape[0], size=n_show, replace=False)
    for ax, result, title in ((axes[0], result_as, "Avellaneda-Stoikov"), (axes[1], result_naive, "Naive symmetric")):
        for i in idx:
            ax.plot(t, result["inventory"][i], color="#1f6f8b", alpha=0.4, linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Inventory (units)")
    fig.suptitle("Sample inventory paths: inventory-skewed vs. symmetric quoting")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pnl_distribution(pnl_as: np.ndarray, pnl_naive: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(min(pnl_as.min(), pnl_naive.min()), max(pnl_as.max(), pnl_naive.max()), 60)
    ax.hist(pnl_naive, bins=bins, alpha=0.5, color="#c1440e", label="Naive symmetric", density=True)
    ax.hist(pnl_as, bins=bins, alpha=0.5, color="#1f6f8b", label="Avellaneda-Stoikov", density=True)
    ax.set_xlabel("Terminal P&L (mark-to-market)")
    ax.set_ylabel("Density")
    ax.set_title("Terminal P&L distribution: inventory-skewed vs. symmetric quoting")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_efficient_frontier(mean_pnls, std_pnls, gammas, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(std_pnls, mean_pnls, c=gammas, cmap="viridis", s=60, zorder=3)
    ax.plot(std_pnls, mean_pnls, color="gray", alpha=0.5, linewidth=1, zorder=2)
    fig.colorbar(sc, ax=ax, label="Risk aversion (gamma)")
    ax.set_xlabel("P&L standard deviation (risk)")
    ax.set_ylabel("Mean P&L (return)")
    ax.set_title("Risk-aversion sweep: mean P&L vs. P&L risk")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spread_vs_sigma(sigmas, spreads, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sigmas, spreads, "o-", color="#1f6f8b")
    ax.set_xlabel("Volatility (sigma)")
    ax.set_ylabel("Optimal total spread at t=0")
    ax.set_title("Sanity check: spread widens with volatility")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
