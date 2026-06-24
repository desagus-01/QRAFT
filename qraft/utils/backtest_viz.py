from __future__ import annotations

from typing import Sequence

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
