from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_simulation_results_on_ax(
    y_paths: NDArray[np.floating],
    *,
    max_paths: int = 30,
    quantiles=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    integrate: bool = False,
    logP0: float = 0.0,
    title: str | None = None,
    ax=None,
):
    y = np.asarray(y_paths, dtype=float)
    if y.ndim != 2:
        raise ValueError("y_paths must be a 2D array (n_paths, horizon).")

    n_paths, _ = y.shape

    if integrate:
        y_cum = logP0 + np.cumsum(y, axis=1)
        y_plot = np.concatenate([np.full((n_paths, 1), logP0), y_cum], axis=1)
        y_label = "Cumulative log-price (logP0 + cumsum)"
    else:
        y_plot = np.concatenate([np.full((n_paths, 1), np.nan), y], axis=1)
        y_label = "Simulated values"

    x = np.arange(y_plot.shape[1])
    rng = np.random.default_rng(0)
    k = min(max_paths, n_paths)
    idx = (
        rng.choice(n_paths, size=k, replace=False)
        if n_paths > k
        else np.arange(n_paths)
    )

    qs = np.array(quantiles, dtype=float)
    qvals = np.empty((len(qs), y_plot.shape[1]), dtype=float)
    qvals[:, 0] = y_plot[0, 0]
    qvals[:, 1:] = np.quantile(y_plot[:, 1:], qs, axis=0)

    if not np.any(np.isclose(qs, 0.50)):
        raise ValueError("quantiles must include 0.50 for the median.")
    med_i = int(np.where(np.isclose(qs, 0.50))[0][0])

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    for lo, hi in [(0.25, 0.75), (0.05, 0.95), (0.01, 0.99)]:
        if np.any(np.isclose(qs, lo)) and np.any(np.isclose(qs, hi)):
            lo_i = int(np.where(np.isclose(qs, lo))[0][0])
            hi_i = int(np.where(np.isclose(qs, hi))[0][0])
            ax.fill_between(x, qvals[lo_i], qvals[hi_i], alpha=0.20, linewidth=0)

    ax.plot(x, qvals[med_i], linewidth=2, label="Median")
    for j in idx:
        ax.plot(x, y_plot[j], alpha=0.35, linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.set_title(
        title or ("Simulated paths (integrated)" if integrate else "Simulated paths")
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax.figure
