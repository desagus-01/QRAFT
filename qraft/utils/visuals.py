# INFO: Below made with AI, cba with visuals - hate it
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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


def _unique_dates_np(dates: pl.DataFrame) -> np.ndarray:
    return dates.select(pl.col("date")).unique(maintain_order=True).to_numpy().ravel()


def plt_prob_shift(
    prior_prob: np.ndarray,
    post_prob: np.ndarray,
    dates: pl.DataFrame,
    k_top: int = 30,
    # WHY panel:
    scenarios: pl.DataFrame | None = None,
    why_assets: list[str] | None = None,
    zscore_assets: bool = True,  # standardise so assets share y-axis
    highlight_k: int = 10,  # how many up + down dates to highlight
    highlight_alpha: float = 0.10,  # transparency of highlight spans
) -> None:
    """
    Panels:
      1) posterior prob + prior baseline, with mover markers on posterior
      2) top-k |Δp| bars (red 0-line)
      3) WHY: selected assets on one axis (optionally z-scored), with up/down mover date highlights
    """
    x = _unique_dates_np(dates)

    if len(prior_prob) != len(post_prob) or len(x) != len(post_prob):
        raise ValueError(
            f"Length mismatch: len(x)={len(x)}, len(prior)={len(prior_prob)}, len(post)={len(post_prob)}"
        )

    eps = 1e-18
    df = (
        pl.DataFrame({"date": x, "p0": prior_prob, "p1": post_prob})
        .with_columns(
            (pl.col("p1") - pl.col("p0")).alias("dp"),
            ((pl.col("p1") + eps).log() - (pl.col("p0") + eps).log()).alias(
                "log_ratio"
            ),
        )
        .with_row_index("idx")
    )

    # --- stats ---
    tv = 0.5 * float(df.select(pl.col("dp").abs().sum()).item())
    kl = float(df.select((pl.col("p1") * pl.col("log_ratio")).sum()).item())
    ess0 = 1.0 / float(df.select((pl.col("p0") ** 2).sum()).item())
    ess1 = 1.0 / float(df.select((pl.col("p1") ** 2).sum()).item())

    # --- top movers by |Δp| for bars + markers ---
    top = (
        df.select("idx", "date", "dp")
        .with_columns(pl.col("dp").abs().alias("abs_dp"))
        .sort("abs_dp", descending=True)
        .head(k_top)
        .sort("date")
    )
    top_idx = top.get_column("idx").to_numpy()
    top_dates = top.get_column("date").to_numpy()
    top_dp = top.get_column("dp").to_numpy()

    # --- select up/down highlight dates (separately) ---
    movers_sorted = (
        df.select("date", "dp")
        .with_columns(pl.col("dp").abs().alias("abs_dp"))
        .sort("abs_dp", descending=True)
    )
    up_dates = (
        movers_sorted.filter(pl.col("dp") > 0)
        .head(highlight_k)
        .get_column("date")
        .to_numpy()
    )
    down_dates = (
        movers_sorted.filter(pl.col("dp") < 0)
        .head(highlight_k)
        .get_column("date")
        .to_numpy()
    )

    # --- layout: add WHY panel only if inputs provided ---
    use_why = scenarios is not None and why_assets
    nrows = 3 if use_why else 2

    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(12, 8.5 if use_why else 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": ([1.35, 1.0, 1.25] if use_why else [1.35, 1.0])},
    )
    if nrows == 2:
        ax_prob, ax_bar = axs
        ax_why = None
    else:
        ax_prob, ax_bar, ax_why = axs

    # ---------------- Panel 1: posterior + prior baseline + mover markers ----------------
    p1 = df.get_column("p1").to_numpy()
    p0 = df.get_column("p0").to_numpy()

    ax_prob.plot(x, p1, label="posterior", linewidth=1.4)
    ax_prob.plot(x, p0, label="prior", linewidth=1.0)
    ax_prob.scatter(top_dates, p1[top_idx], s=22, zorder=3)

    ax_prob.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 3))
    ax_prob.set_ylabel("Scenario prob")
    ax_prob.legend(loc="upper right")

    stats_txt = f"TV={tv:.5f}  KL={kl:.6f}  ESS0={ess0:.1f}  ESS1={ess1:.1f}"
    ax_prob.text(
        0.01,
        0.02,
        stats_txt,
        transform=ax_prob.transAxes,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.9),
        fontsize=10,
    )

    # ---------------- Panel 2: top-k Δp bars ----------------
    colors = np.where(top_dp >= 0, "C0", "C3")
    ax_bar.bar(top_dates, top_dp, color=colors)
    ax_bar.axhline(0.0, linewidth=1.2, color="red")
    ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 3))
    ax_bar.set_ylabel(f"Top {k_top} Δp")

    # ---------------- Panel 3: WHY (assets together) ----------------
    if use_why and ax_why is not None:
        sc = scenarios.with_columns(pl.Series("date", x))

        # highlight bands first (so lines draw on top)
        for d in up_dates:
            ax_why.axvspan(
                d, d, color="C0", alpha=highlight_alpha
            )  # very thin span (acts like vline)
        for d in down_dates:
            ax_why.axvspan(d, d, color="C3", alpha=highlight_alpha)

        # plot assets as (optionally) z-scored lines
        for a in why_assets:
            y = sc.get_column(a).to_numpy()

            if zscore_assets:
                mu = float(np.nanmean(y))
                sd = float(np.nanstd(y))
                y_plot = (y - mu) / (sd if sd > 0 else 1.0)
            else:
                y_plot = y

            ax_why.plot(x, y_plot, label=a, linewidth=1.1)

        ax_why.axhline(0.0, linewidth=1.0, color="red")
        ax_why.set_ylabel("Assets (z-score)" if zscore_assets else "Asset level")
        ax_why.legend(loc="upper left", ncols=min(3, len(why_assets)))

        # Add a tiny note for what the highlights mean
        ax_why.text(
            0.01,
            0.02,
            f"Highlights: top {highlight_k} upweighted (C0) + top {highlight_k} downweighted (C3) dates by |Δp|",
            transform=ax_why.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            alpha=0.85,
        )

    # --- date formatting ---
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()


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
