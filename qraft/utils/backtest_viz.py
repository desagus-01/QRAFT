from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.figure import Figure

from qraft.backtest.execution import BacktestResult
from qraft.backtest.metrics import PerformanceSummary
from qraft.core.metrics import drawdown_curve, returns_from_nav


def _to_dates(dates: Sequence) -> list[float]:
    return mdates.date2num(list(dates))


def summary_stats(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252,
) -> dict[str, float]:
    return PerformanceSummary.from_backtest(
        result,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    ).to_dict()


def plot_nav(
    results: BacktestResult | Sequence[BacktestResult],
    *,
    labels: str | Sequence[str] | None = None,
    log_scale: bool = False,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252,
    figsize: tuple[float, float] = (12, 5),
    title: str = "Backtest NAV",
) -> Figure:
    if isinstance(results, BacktestResult):
        results = [results]

    if labels is None:
        labels = [r.policy_name for r in results]
    elif isinstance(labels, str):
        labels = [labels]
    if len(labels) != len(results):
        labels = [r.policy_name for r in results]

    fig, ax = plt.subplots(figsize=figsize)
    for r, lbl in zip(results, labels):
        dates = _to_dates(r.nav_dates)
        ax.plot(dates, np.asarray(r.nav, dtype=float), label=lbl, linewidth=1.5)

    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.2f}"))
    ax.set_ylabel("NAV")
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    if len(results) == 1:
        r = results[0]
        s = summary_stats(r, risk_free_rate, periods_per_year)
        text = (
            f"Ann. Return: {s['annualised_return']:.2%}  "
            f"Vol: {s['annualised_vol']:.2%}  "
            f"Sharpe: {s['sharpe']:.2f}  "
            f"Max DD: {s['max_drawdown']:.2%}  "
            f"Calmar: {s['calmar']:.2f}"
        )
        ax.text(
            0.01,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
            ),
        )

    fig.tight_layout()
    return fig


def plot_drawdown(
    result: BacktestResult,
    *,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
) -> Figure:
    nav = np.asarray(result.nav, dtype=float)
    dd = drawdown_curve(nav)
    dates = _to_dates(result.nav_dates)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(dates, 0.0, dd, color="crimson", alpha=0.5, linewidth=0)
    ax.plot(dates, dd, color="crimson", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Drawdown")
    ax.set_title(title or f"Drawdown — {result.policy_name}")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_weights(
    result: BacktestResult,
    *,
    max_assets: int = 10,
    include_cash: bool = True,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
) -> Figure:
    dates = _to_dates(result.period_execution_bars)
    w = result.period_weights
    if not include_cash:
        w = w[:, :-1]
    n_assets_total = w.shape[1]

    if n_assets_total <= max_assets + 1:
        weights_to_plot = w
        labels = list(result.asset_order)
        if include_cash:
            labels.append("Cash")
    else:
        mean_weights = np.mean(w, axis=0)
        top_idx = np.argsort(mean_weights)[-max_assets:]
        remaining_idx = [i for i in range(n_assets_total) if i not in top_idx]
        weights_to_plot = np.column_stack(
            [w[:, top_idx], np.sum(w[:, remaining_idx], axis=1)]
        )
        all_labels = list(result.asset_order) + (["Cash"] if include_cash else [])
        labels = [all_labels[i] for i in sorted(top_idx)] + ["Others"]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, weights_to_plot.shape[1]))
    ax.stackplot(dates, weights_to_plot.T, labels=labels, colors=colors, alpha=0.85)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Allocation")
    ax.set_title(title or f"Portfolio Weights Over Time — {result.policy_name}")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.legend(loc="upper left", frameon=True, fancybox=False, fontsize=8, ncol=2)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_returns_hist(
    result: BacktestResult,
    *,
    bins: int = 40,
    density: bool = True,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
) -> Figure:
    returns = np.asarray(result.period_returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.hist(
        returns,
        bins=bins,
        density=density,
        alpha=0.6,
        color="steelblue",
        edgecolor="white",
    )
    ax.axvline(0.0, color="red", linewidth=1.0, linestyle="--")
    ax.axvline(
        np.mean(returns),
        color="black",
        linewidth=1.5,
        label=f"Mean: {np.mean(returns):.2%}",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Period Return")
    ax.set_ylabel("Density")
    ax.set_title(f"Return Distribution — {result.policy_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sorted_rets = np.sort(returns)
    pct = np.arange(1, len(sorted_rets) + 1) / len(sorted_rets) * 100
    ax.plot(pct, sorted_rets, color="steelblue", linewidth=1.5)
    ax.axhline(0.0, color="red", linewidth=0.8, linestyle="--")
    ax.axhline(
        np.mean(returns),
        color="black",
        linewidth=1.2,
        label=f"Mean: {np.mean(returns):.2%}",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Period Return")
    ax.set_title("Ordered Returns (CDF)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    stats_text = (
        f"Mean: {np.mean(returns):.2%}\n"
        f"Std:  {np.std(returns, ddof=1):.2%}\n"
        f"Skew: {float(np.mean((returns - np.mean(returns)) ** 3) / np.std(returns, ddof=1) ** 3):.2f}\n"
        f"Kurt: {float(np.mean((returns - np.mean(returns)) ** 4) / np.std(returns, ddof=1) ** 4 - 3):.2f}\n"
        f"Min:  {np.min(returns):.2%}\n"
        f"Max:  {np.max(returns):.2%}\n"
        f"Hit:  {np.mean(returns > 0):.1%}"
    )
    axes[1].text(
        0.97,
        0.97,
        stats_text,
        transform=axes[1].transAxes,
        va="top",
        ha="right",
        fontsize=8,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"
        ),
    )

    fig.suptitle(title or "Period Return Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_rolling_metrics(
    result: BacktestResult,
    *,
    window: int = 252,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
) -> Figure:
    nav = np.asarray(result.nav, dtype=float)
    dates = _to_dates(result.nav_dates)
    returns = returns_from_nav(nav)

    rolling_mean = np.full(len(nav), np.nan)
    rolling_std = np.full(len(nav), np.nan)
    for i in range(window, len(nav)):
        seg = returns[i - window : i]
        rolling_mean[i] = np.mean(seg)
        rolling_std[i] = np.std(seg, ddof=1)

    rolling_ann_ret = rolling_mean * periods_per_year
    rolling_ann_vol = rolling_std * np.sqrt(periods_per_year)
    rolling_sharpe = (
        (rolling_mean - risk_free_rate / periods_per_year)
        / rolling_std
        * np.sqrt(periods_per_year)
    )

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    ax.plot(dates, rolling_ann_ret, color="forestgreen", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel(f"{window}-day Ann. Return")
    ax.set_title(title or f"Rolling Metrics — {result.policy_name}")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(dates, rolling_ann_vol, color="steelblue", linewidth=1.2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel(f"{window}-day Ann. Volatility")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(dates, rolling_sharpe, color="darkorange", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel(f"{window}-day Sharpe")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_turnover_and_costs(
    result: BacktestResult,
    *,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> Figure:
    dates = _to_dates(result.period_execution_bars)
    turnover = np.asarray(result.period_turnovers, dtype=float)
    costs = np.asarray(result.period_costs, dtype=float)

    fig, ax1 = plt.subplots(figsize=figsize)

    color_to = "steelblue"
    ax1.bar(
        dates,
        turnover,
        color=color_to,
        alpha=0.6,
        width=0.7,
        label="Turnover (one-way)",
    )
    ax1.set_ylabel("Turnover", color=color_to)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.tick_params(axis="y", labelcolor=color_to)

    ax2 = ax1.twinx()
    color_cost = "crimson"
    ax2.plot(
        dates,
        costs,
        color=color_cost,
        marker="o",
        linewidth=1.2,
        markersize=4,
        label="Costs",
    )
    ax2.set_ylabel("Costs (NAV units)", color=color_cost)
    ax2.tick_params(axis="y", labelcolor=color_cost)

    ax1.set_xlabel("Date")
    ax1.set_title(title or f"Turnover & Costs — {result.policy_name}")
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=True,
        fancybox=False,
    )

    fig.tight_layout()
    return fig


def plot_comparison(
    results: Sequence[BacktestResult],
    *,
    labels: Sequence[str] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252,
    figsize: tuple[float, float] = (12, 10),
    title: str = "Strategy Comparison",
) -> Figure:
    if labels is None:
        labels = [r.policy_name for r in results]
    if len(labels) != len(results):
        labels = [r.policy_name for r in results]

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    ax = axes[0, 0]
    for r, lbl in zip(results, labels):
        dates = _to_dates(r.nav_dates)
        ax.plot(dates, np.asarray(r.nav, dtype=float), label=lbl, linewidth=1.2)
    ax.set_title("NAV")
    ax.set_ylabel("NAV")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.legend(fontsize=7, frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for r, lbl in zip(results, labels):
        nav = np.asarray(r.nav, dtype=float)
        dd = drawdown_curve(nav)
        dates = _to_dates(r.nav_dates)
        ax.plot(dates, dd, label=lbl, linewidth=1.2)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.legend(fontsize=7, frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for r, lbl in zip(results, labels):
        ret = np.asarray(r.period_returns, dtype=float)
        ret = ret[np.isfinite(ret)]
        if len(ret) > 0:
            ax.hist(ret, bins=30, alpha=0.4, label=lbl, density=True)
    ax.set_title("Return Distribution")
    ax.set_xlabel("Period Return")
    ax.set_ylabel("Density")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=7, frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    bar_data = {}
    for r, lbl in zip(results, labels):
        to = np.asarray(r.period_turnovers, dtype=float)
        to = to[np.isfinite(to)]
        bar_data[lbl] = np.mean(to) if len(to) > 0 else 0.0
    x_pos = np.arange(len(bar_data))
    ax.bar(x_pos, list(bar_data.values()), color="steelblue", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(bar_data.keys()), fontsize=8)
    ax.set_title("Average Turnover")
    ax.set_ylabel("One-way Turnover")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2, 0]
    for r, lbl in zip(results, labels):
        c = np.asarray(r.period_costs, dtype=float)
        c = c[np.isfinite(c)]
        ax.plot(_to_dates(r.period_execution_bars), c, label=lbl, linewidth=1.2)
    ax.set_title("Trading Costs Over Time")
    ax.set_ylabel("Costs (NAV units)")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.legend(fontsize=7, frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.axis("off")
    stats_list = []
    for r in results:
        s = summary_stats(r, risk_free_rate, periods_per_year)
        stats_list.append(s)
    header = f"{'Strategy':<18} {'Ann.Ret':>8} {'Vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'TO':>7}"
    ax.text(
        0.05,
        0.95,
        header,
        transform=ax.transAxes,
        fontsize=8,
        family="monospace",
        va="top",
    )
    for i, (lbl, s) in enumerate(zip(labels, stats_list)):
        line = f"{lbl:<18} {s['annualised_return']:>7.2%} {s['annualised_vol']:>7.2%} {s['sharpe']:>6.2f} {s['max_drawdown']:>7.2%} {s['calmar']:>6.2f} {s['avg_turnover']:>6.2%}"
        ax.text(
            0.05,
            0.90 - i * 0.045,
            line,
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            va="top",
        )

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_backtest_dashboard(
    result: BacktestResult,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252,
    figsize: tuple[float, float] = (14, 10),
    title: str | None = None,
) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    nav = np.asarray(result.nav, dtype=float)
    dates_nav = _to_dates(result.nav_dates)
    dd = drawdown_curve(nav)
    returns = returns_from_nav(nav)

    ax = axes[0, 0]
    ax.plot(dates_nav, nav, linewidth=1.5, color="steelblue")
    ax.set_title("NAV")
    ax.set_ylabel("NAV")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.fill_between(dates_nav, 0.0, dd, color="crimson", alpha=0.4)
    ax.plot(dates_nav, dd, color="crimson", linewidth=1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Drawdown")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    dates_ex = _to_dates(result.period_execution_bars)
    to = np.asarray(result.period_turnovers, dtype=float)
    costs = np.asarray(result.period_costs, dtype=float)
    ax.bar(dates_ex, to, color="steelblue", alpha=0.5, label="Turnover")
    ax_twin = ax.twinx()
    ax_twin.plot(
        dates_ex,
        costs,
        color="crimson",
        marker="o",
        linewidth=1.0,
        markersize=3,
        label="Costs",
    )
    ax.set_title("Turnover & Costs")
    ax.set_ylabel("Turnover")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_twin.set_ylabel("Costs")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    rets_clean = returns[np.isfinite(returns)]
    ax.hist(
        rets_clean,
        bins=50,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="white",
    )
    ax.axvline(0.0, color="red", linewidth=1.0, linestyle="--")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Return")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    perf_text_lines = []
    s = summary_stats(result, risk_free_rate, periods_per_year)
    for k, v in s.items():
        if isinstance(v, float):
            if abs(v) < 1.0:
                perf_text_lines.append(f"{k:20s}: {v:>8.4f}")
            else:
                perf_text_lines.append(f"{k:20s}: {v:>8.2f}")
        else:
            perf_text_lines.append(f"{k:20s}: {v!s:>8}")
    ax.text(
        0.1,
        0.95,
        "\n".join(perf_text_lines),
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        va="top",
    )
    ax.set_title("Summary Statistics")
    ax.axis("off")

    ax = axes[1, 2]
    w = result.period_weights
    n_assets = len(result.asset_order)
    if w.shape[1] <= 12:
        weights_to_plot = w[:, :-1]
        labels_w = result.asset_order
    else:
        mean_w = np.mean(w[:, :-1], axis=0)
        top_idx = np.argsort(mean_w)[-8:]
        remaining_idx = [i for i in range(n_assets) if i not in top_idx]
        weights_to_plot = np.column_stack(
            [w[:, top_idx], np.sum(w[:, remaining_idx], axis=1)]
        )
        labels_w = [result.asset_order[i] for i in sorted(top_idx)] + ["Others"]
    colors = plt.cm.tab20(np.linspace(0, 1, weights_to_plot.shape[1]))
    ax.stackplot(dates_ex, weights_to_plot.T, labels=labels_w, colors=colors, alpha=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Allocation Over Time")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.legend(loc="upper left", fontsize=6, frameon=True, ncol=2)

    fig.suptitle(
        title or f"Backtest Dashboard — {result.policy_name}", fontsize=14, y=1.01
    )
    fig.tight_layout()
    return fig


def plot_walk_forward_report(
    *,
    folds: Sequence[Any],
    folds_df: Any,
    selection_counts_df: Any,
    oos_summary: PerformanceSummary | None,
    oos_nav_dates: Sequence[datetime],
    oos_nav: Sequence[float],
    n_trials: int,
    deflated_sharpe: float | None,
    pbo: float | None,
) -> Figure:
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    dates = list(oos_nav_dates)
    nav = np.asarray(oos_nav, dtype=float)

    ax = axes[0, 0]
    if nav.size:
        ax.plot(dates, nav, color="steelblue", linewidth=1.6)
        for fold_result in folds:
            ax.axvspan(
                fold_result.fold.test[0],
                fold_result.fold.test[1],
                color="steelblue",
                alpha=0.06,
            )
    ax.set_title("Stitched OOS NAV")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if nav.size:
        peak = np.maximum.accumulate(nav)
        drawdown = nav / peak - 1.0
        ax.fill_between(dates, 0.0, drawdown, color="crimson", alpha=0.35)
        ax.plot(dates, drawdown, color="crimson", linewidth=1.0)
    ax.set_title("Stitched OOS Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    fold_ids = np.arange(len(folds))
    test_returns = _column_or_nan(folds_df, "test_total_return", len(folds))
    colors = ["forestgreen" if r >= 0 else "crimson" for r in test_returns]
    ax.bar(fold_ids, test_returns, color=colors, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("OOS Return By Fold")
    ax.set_xlabel("Fold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    train_sharpe = _column_or_nan(folds_df, "train_sharpe", len(folds))
    test_sharpe = _column_or_nan(folds_df, "test_sharpe", len(folds))
    ax.plot(fold_ids, train_sharpe, marker="o", label="Train selected")
    ax.plot(fold_ids, test_sharpe, marker="o", label="OOS")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Sharpe By Fold")
    ax.set_xlabel("Fold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    sharpe_decay = _column_or_nan(folds_df, "sharpe_decay", len(folds))
    colors = ["forestgreen" if d >= 0 else "crimson" for d in sharpe_decay]
    ax.bar(fold_ids, sharpe_decay, color=colors, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("OOS Sharpe Decay")
    ax.set_xlabel("Fold")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2, 1]
    if not selection_counts_df.is_empty():
        labels = selection_counts_df["selected_params"].to_list()
        values = selection_counts_df["n_folds"].to_list()
        y = np.arange(len(labels))
        ax.barh(y, values, color="slateblue", alpha=0.75)
        ax.set_yticks(y, [_truncate_label(label) for label in labels])
        ax.invert_yaxis()
    ax.set_title("Selected Parameter Stability")
    ax.set_xlabel("Folds selected")
    ax.grid(True, axis="x", alpha=0.3)

    _plot_diagnostics_bar(
        axes[3, 0],
        title="Multiple-Testing Diagnostics",
        n_trials=n_trials,
        deflated_sharpe=deflated_sharpe,
        pbo=pbo,
    )

    ax = axes[3, 1]
    ax.axis("off")
    diagnostic_text = [
        "Robustness read-through",
        f"Trials tested: {n_trials}",
        f"Deflated Sharpe: {_format_optional_float(deflated_sharpe)}",
        f"PBO: {_format_optional_pct(pbo)}",
    ]
    if deflated_sharpe is not None:
        diagnostic_text.append(
            "DSR adjusts the observed OOS Sharpe for multiple trials and non-normality."
        )
    if pbo is not None:
        diagnostic_text.append(
            "PBO estimates how often selection would pick a strategy that underperforms OOS."
        )
    ax.text(
        0.0,
        1.0,
        "\n".join(diagnostic_text),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        linespacing=1.5,
    )

    for ax in axes.flat[:2]:
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )

    title = "Walk-Forward OOS Report"
    if oos_summary is not None:
        title += (
            f" | return {oos_summary.total_return:.1%}, "
            f"Sharpe {oos_summary.sharpe:.2f}, "
            f"max DD {oos_summary.max_drawdown:.1%}"
        )
    if deflated_sharpe is not None:
        title += f" | DSR {deflated_sharpe:.2f}"
    if pbo is not None:
        title += f" | PBO {pbo:.1%}"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_combinatorial_report(
    *,
    paths: Sequence[PerformanceSummary],
    path_sharpes: Sequence[float],
    path_returns: Sequence[Sequence[float]],
    n_paths: int,
    n_groups: int,
    n_test_groups: int,
    n_trials: int,
    median_sharpe: float,
    sharpe_iqr: tuple[float, float],
    worst_sharpe: float,
    deflated_sharpe: float | None,
    pbo: float | None,
) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sharpes = np.asarray(path_sharpes, dtype=float)
    finite_sharpes = sharpes[np.isfinite(sharpes)]

    ax = axes[0, 0]
    if finite_sharpes.size:
        ax.hist(finite_sharpes, bins="auto", color="steelblue", alpha=0.75)
        ax.axvline(
            median_sharpe,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"Median {median_sharpe:.2f}",
        )
        ax.axvline(0.0, color="crimson", linewidth=0.9, alpha=0.8)
        ax.legend(fontsize=8)
    ax.set_title("CPCV Sharpe Distribution")
    ax.set_xlabel("Path Sharpe")
    ax.set_ylabel("Paths")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    for returns in path_returns:
        nav = _path_nav(returns)
        if nav.size:
            ax.plot(nav, color="steelblue", linewidth=1.0, alpha=0.3)
    if paths:
        terminal = np.array([p.total_return + 1.0 for p in paths], dtype=float)
        if np.isfinite(terminal).any():
            ax.axhline(
                float(np.nanmedian(terminal)),
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Median terminal NAV",
            )
            ax.legend(fontsize=8)
    ax.set_title("CPCV Path Fan")
    ax.set_xlabel("Path period")
    ax.set_ylabel("Normalized NAV")
    ax.grid(True, alpha=0.3)

    _plot_diagnostics_bar(
        axes[1, 0],
        title="PBO / DSR Diagnostics",
        n_trials=n_trials,
        deflated_sharpe=deflated_sharpe,
        pbo=pbo,
    )

    ax = axes[1, 1]
    ax.axis("off")
    lo, hi = sharpe_iqr
    diagnostic_text = [
        "CPCV robustness read-through",
        f"Paths: {n_paths}",
        f"Groups: {n_groups}",
        f"Test groups per fold: {n_test_groups}",
        f"Trials tested: {n_trials}",
        f"Median Sharpe: {_format_optional_float(median_sharpe)}",
        f"Sharpe IQR: {_format_optional_float(lo)} to {_format_optional_float(hi)}",
        f"Worst Sharpe: {_format_optional_float(worst_sharpe)}",
        f"Deflated Sharpe: {_format_optional_float(deflated_sharpe)}",
        f"PBO: {_format_optional_pct(pbo)}",
    ]
    if deflated_sharpe is not None:
        diagnostic_text.append(
            "DSR adjusts the observed path Sharpe for multiple trials and non-normality."
        )
    if pbo is not None:
        diagnostic_text.append(
            "PBO estimates how often selection would pick a strategy that underperforms OOS."
        )
    ax.text(
        0.0,
        1.0,
        "\n".join(diagnostic_text),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        linespacing=1.5,
    )

    title = (
        f"CPCV Report | paths {n_paths}, "
        f"median Sharpe {_format_optional_float(median_sharpe)}"
    )
    if deflated_sharpe is not None:
        title += f" | DSR {deflated_sharpe:.2f}"
    if pbo is not None:
        title += f" | PBO {pbo:.1%}"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def _plot_diagnostics_bar(
    ax: Any,
    *,
    title: str,
    n_trials: int,
    deflated_sharpe: float | None,
    pbo: float | None,
) -> None:
    diagnostic_labels = ["Deflated\nSharpe", "PBO", "Trials"]
    diagnostic_values = [
        deflated_sharpe if deflated_sharpe is not None else np.nan,
        pbo if pbo is not None else np.nan,
        float(n_trials),
    ]
    colors = ["forestgreen", "crimson", "steelblue"]
    bars = ax.bar(diagnostic_labels, diagnostic_values, color=colors, alpha=0.75)
    for bar, value, label in zip(
        bars, diagnostic_values, diagnostic_labels, strict=True
    ):
        if np.isfinite(value):
            text = f"{value:.1%}" if label == "PBO" else f"{value:.2f}"
            if label == "Trials":
                text = f"{int(value)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def _column_or_nan(frame: Any, name: str, length: int) -> np.ndarray:
    if name not in frame.columns:
        return np.full(length, np.nan)
    return np.array(frame[name].to_list(), dtype=float)


def _truncate_label(label: str, max_len: int = 42) -> str:
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 3]}..."


def _path_nav(returns: Sequence[float]) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.array([], dtype=float)
    return np.concatenate(([1.0], np.cumprod(1.0 + returns)))


def _format_optional_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def _format_optional_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1%}"
