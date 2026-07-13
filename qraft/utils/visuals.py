# INFO: Below made with AI, cba with visuals - hate it
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from qraft.forecast.plotting import plot_simulation_results_on_ax


def as_mpl_dates(dates: Any) -> list[float]:
    """Convert datetime-like values to Matplotlib's numeric date representation."""
    return mdates.date2num(list(dates))


def format_date_axis(ax: Any, *, rotate: int = 0) -> None:
    """Apply the project-standard date locator and formatter to an axis."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if rotate:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate)
            label.set_horizontalalignment("right")


def plot_hist_compare(base, cma, column):
    plt.figure(figsize=(7, 4))

    plt.hist(base[column].to_numpy(), bins=40, alpha=0.5, density=True, label="Base")
    plt.hist(cma[column].to_numpy(), bins=40, alpha=0.5, density=True, label="CMA")

    plt.title(f"Distribution Comparison: {column}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_compare(base, cma, x, y):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].scatter(base[x], base[y], s=5, alpha=0.3)
    axs[0].set_title("Base")
    axs[0].set_xlabel(x)
    axs[0].set_ylabel(y)

    axs[1].scatter(cma[x], cma[y], s=5, alpha=0.3)
    axs[1].set_title("CMA")
    axs[1].set_xlabel(x)
    axs[1].set_ylabel(y)

    plt.suptitle(f"Dependence Structure: {x} vs {y}")
    plt.tight_layout()
    plt.show()


def _to_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def plot_residual_acf_pacf(
    asset: str,
    residuals: Any,
    *,
    lags: int | None = 40,
    alpha: float = 0.05,
    pacf_method: str = "ywm",
    save_path: str | Path | None = None,
    show: bool = True,
    title_suffix: str | None = None,
) -> plt.Figure:
    """
    2x2 grid:
      [ACF(resid),  PACF(resid)]
      [ACF(resid²), PACF(resid²)]
    """
    resid = _to_1d_float_array(residuals)
    resid2 = resid**2

    base_title = asset if not title_suffix else f"{asset} | {title_suffix}"

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    plot_acf(resid, lags=lags, alpha=alpha, ax=axes[0, 0])
    axes[0, 0].set_title(f"ACF: {base_title} | residuals")

    plot_pacf(resid, lags=lags, alpha=alpha, method=pacf_method, ax=axes[0, 1])
    axes[0, 1].set_title(f"PACF: {base_title} | residuals")

    plot_acf(resid2, lags=lags, alpha=alpha, ax=axes[1, 0])
    axes[1, 0].set_title(f"ACF: {base_title} | residuals**2")

    plot_pacf(resid2, lags=lags, alpha=alpha, method=pacf_method, ax=axes[1, 1])
    axes[1, 1].set_title(f"PACF: {base_title} | residuals**2")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_simulation_results(
    y_paths: np.ndarray,
    *,
    max_paths: int = 30,
    quantiles=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    integrate: bool = False,
    logP0: float = 0.0,
    title: str | None = None,
    ax=None,
):
    """
    Plot simulation paths and a fan chart.

    Parameters
    ----------
    y_paths : array, shape (n_paths, horizon)
        Output from simulate_asset_paths.
    max_paths : int
        How many sample paths to draw as lines.
    quantiles : tuple
        Quantiles for the fan chart (must include 0.5 for the median).
    integrate : bool
        If True, plots cumulative sum along time (useful for log-price diffs).
    logP0 : float
        Starting log price if integrate=True (e.g., np.log(last_price)).
    title : str | None
        Figure title.
    """
    fig = plot_simulation_results_on_ax(
        y_paths,
        max_paths=max_paths,
        quantiles=quantiles,
        integrate=integrate,
        logP0=logP0,
        title=title,
        ax=ax,
    )
    fig.tight_layout()
    return fig


def plot_effective_bets(effective_bets_result) -> None:
    factor_contributions = effective_bets_result.factor_contributions
    enb = effective_bets_result.effective_bets

    if not factor_contributions:
        raise ValueError("factor_contributions is empty, nothing to plot.")

    df = pl.DataFrame(
        {
            "bet": list(factor_contributions.keys()),
            "contribution": list(factor_contributions.values()),
        }
    )

    df = (
        df.with_columns((pl.col("contribution") * 100).alias("pct"))
        .sort("contribution", descending=True)
        .with_columns(pl.col("pct").cum_sum().alias("cumulative_pct"))
    )

    bets = df["bet"].to_list()
    pct_list = df["pct"].to_list()
    cumulative_list = df["cumulative_pct"].to_list()

    # 1) Contribution bar chart
    plt.figure(figsize=(8, 4.8))
    positions = range(len(bets))
    plt.bar(positions, pct_list)
    plt.ylabel("Risk contribution (%)")
    plt.xlabel("Bet")
    plt.title("Effective Bets: Contribution by Bet")
    plt.xticks(list(positions), bets, rotation=0)

    y_offset = max(pct_list) * 0.02 if pct_list else 0.0
    for i, v in enumerate(pct_list):
        plt.text(i, v + y_offset, f"{v:.1f}%", ha="center")

    plt.ylim(0, max(pct_list) * 1.18 if pct_list else 1)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 2) Concentration curve
    plt.figure(figsize=(8, 4.8))
    x_positions = list(range(1, len(bets) + 1))
    plt.plot(x_positions, cumulative_list, marker="o")
    plt.xticks(x_positions, bets, rotation=0)
    plt.ylabel("Cumulative risk contribution (%)")
    plt.xlabel("Ordered bets")
    plt.title("Effective Bets: Concentration Curve")

    for i, v in enumerate(cumulative_list, start=1):
        plt.text(i, v + 2, f"{v:.1f}%", ha="center")

    plt.ylim(0, max(110, max(cumulative_list) * 1.05 if cumulative_list else 1))
    plt.tight_layout()
    plt.show()
    plt.close()

    # 3) ENB vs maximum possible
    max_bets = len(bets)
    labels = ["Effective bets", "Maximum possible"]
    values = [enb, max_bets]

    plt.figure(figsize=(7, 4.8))
    plt.bar(labels, values)
    plt.ylabel("Number of bets")
    plt.title("Effective Bets vs Maximum Possible")

    y_offset = max(values) * 0.02 if values else 0.0
    for i, v in enumerate(values):
        plt.text(
            i,
            v + y_offset,
            f"{v:.2f}" if i == 0 else f"{int(v)}",
            ha="center",
        )

    plt.ylim(0, max(values) * 1.2 if values else 1)
    plt.tight_layout()
    plt.show()
    plt.close()
