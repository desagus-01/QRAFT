"""Backtest result records, summaries, and plotting helpers."""

from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from numpy.typing import NDArray

from qraft.backtest.result.period import BacktestPeriod
from qraft.backtest.result.summary import PerformanceSummary
from qraft.core.metrics import drawdown_curve, returns_from_nav
from qraft.core.weights import target_weights_full
from qraft.utils.visuals import as_mpl_dates, format_date_axis

BacktestWarning = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Portfolio NAV, rebalance periods, costs, warnings, and diagnostics."""

    policy_name: str
    asset_order: list[str]
    nav_dates: list[datetime]
    nav: NDArray[np.floating]
    periods: list[BacktestPeriod]
    warnings_log: list[BacktestWarning] = field(default_factory=list)
    holding_costs: NDArray[np.floating] = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    periods_per_year: float | None = None
    invariance_drops: tuple[Any, ...] = ()

    def window(self, start: datetime, end: datetime) -> "BacktestResult":
        """Return a sub-result over the inclusive date range ``[start, end]``."""
        keep = [start <= d <= end for d in self.nav_dates]
        mask = np.asarray(keep, dtype=bool)
        nav_dates = [d for d, k in zip(self.nav_dates, keep) if k]
        nav = self.nav[mask] if self.nav.size == mask.size else self.nav[:0]
        holding = (
            self.holding_costs[mask]
            if self.holding_costs.size == mask.size
            else self.holding_costs[:0]
        )
        periods = [p for p in self.periods if start <= p.execution_bar <= end]
        warnings = [
            w
            for w in self.warnings_log
            if isinstance(w.get("bar"), datetime) and start <= w["bar"] <= end
        ]
        return BacktestResult(
            policy_name=self.policy_name,
            asset_order=self.asset_order,
            nav_dates=nav_dates,
            nav=nav,
            periods=periods,
            warnings_log=warnings,
            holding_costs=holding,
            periods_per_year=self.periods_per_year,
            invariance_drops=self.invariance_drops,
        )

    def summary(
        self,
        *,
        periods_per_year: float | None = None,
        risk_free_rate: float = 0.0,
        active_only: bool = True,
    ):
        """Return a ``PerformanceSummary`` for this backtest result."""
        from qraft.backtest.result.summary import PerformanceSummary

        return PerformanceSummary.from_backtest(
            self,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            active_only=active_only,
        )

    def plot_nav(self, **kwargs):
        """Return a NAV plot figure for this result."""
        return plot_nav(self, **kwargs)

    def plot_weights(self, **kwargs):
        """Return a portfolio-weights plot figure for this result."""
        return plot_weights(self, **kwargs)

    def plot_drawdown(self, **kwargs):
        """Return a drawdown plot figure for this result."""
        return plot_drawdown(self, **kwargs)

    def plot_turnover_and_costs(self, **kwargs):
        """Return a turnover and transaction-cost plot figure for this result."""
        return plot_turnover_and_costs(self, **kwargs)

    def summary_df(
        self,
        *,
        periods_per_year: float | None = None,
        risk_free_rate: float = 0.0,
        active_only: bool = True,
    ) -> pl.DataFrame:
        """Return one-row performance summary metrics as a Polars DataFrame."""
        summary = self.summary(
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            active_only=active_only,
        ).to_dict()
        return pl.DataFrame([summary])

    def view_activity_df(self) -> pl.DataFrame:
        """Return per-period scenario-view activity and entropy diagnostics."""
        rows = []
        for period in self.periods:
            diag = period.view_diagnostics
            if diag is None:
                rows.append(
                    {
                        "decision_bar": period.decision_bar,
                        "views_active": False,
                        "ens_prior": None,
                        "ens_posterior": None,
                        "ens_ratio": None,
                        "ens_collapsed": None,
                        "n_binding_constraints": 0,
                    }
                )
                continue
            ens_prior = float(diag.ens_prior)
            ens_posterior = float(diag.ens_posterior)
            rows.append(
                {
                    "decision_bar": period.decision_bar,
                    "views_active": True,
                    "ens_prior": ens_prior,
                    "ens_posterior": ens_posterior,
                    "ens_ratio": ens_posterior / ens_prior,
                    "ens_collapsed": bool(diag.ens_collapsed),
                    "n_binding_constraints": sum(
                        1 for constraint in diag.constraints if constraint.get("active")
                    ),
                }
            )
        return pl.DataFrame(rows)

    @property
    def dropped_assets(self) -> tuple[str, ...]:
        """Return sorted assets dropped from any rebalance period."""
        return tuple(
            sorted({asset for p in self.periods for asset in p.dropped_assets})
        )

    @property
    def period_decision_bars(self) -> list[datetime]:
        """Return decision bars for all rebalance periods."""
        return [p.decision_bar for p in self.periods]

    @property
    def period_execution_bars(self) -> list[datetime]:
        """Return execution bars for all rebalance periods."""
        return [p.execution_bar for p in self.periods]

    @property
    def period_returns(self) -> NDArray[np.floating]:
        """Simple return per holding period between rebalances."""
        if not self.periods:
            return np.array([], dtype=float)
        n = len(self.periods)
        start_values = np.empty(n)
        end_values = np.empty(n)
        for i, p in enumerate(self.periods):
            if i == 0:
                start_values[i] = float(self.nav[0])
            else:
                start_values[i] = float(self.periods[i - 1].state_after.portfolio_value)
            end_values[i] = float(p.state_before.portfolio_value)
        return end_values / start_values - 1.0

    @property
    def period_turnovers(self) -> NDArray[np.floating]:
        """One-way turnover fraction at each rebalance."""
        if not self.periods:
            return np.array([], dtype=float)
        turnovers = np.empty(len(self.periods))
        for i, p in enumerate(self.periods):
            pv = float(p.state_before.portfolio_value)
            trade_value = float(
                np.abs(p.executed_share_trades * p.state_before.initial_prices).sum()
            )
            cash_trade = float(p.state_after.cash - p.state_before.cash)
            turnovers[i] = 0.5 * (trade_value + abs(cash_trade)) / pv if pv > 0 else 0.0
        return turnovers

    @property
    def period_costs(self) -> NDArray[np.floating]:
        """Realised **transaction** cost per rebalance (NAV units)."""
        return np.array([p.cost for p in self.periods], dtype=float)

    @property
    def total_transaction_cost(self) -> float:
        """Return total realised transaction cost in NAV units."""
        return float(self.period_costs.sum())

    @property
    def total_holding_cost(self) -> float:
        """Return total holding cost in NAV units."""
        return float(self.holding_costs.sum())

    @property
    def total_cost(self) -> float:
        """Return total realised transaction and holding costs."""
        return self.total_transaction_cost + self.total_holding_cost

    @property
    def period_target_weights_array(self) -> NDArray[np.floating]:
        """Target total weights per period, shape (n_periods, n_assets + 1)."""
        n = len(self.periods)
        n_assets = len(self.asset_order)
        out = np.empty((n, n_assets + 1))
        for i, p in enumerate(self.periods):
            out[i, :-1] = target_weights_full(p.decision, self.asset_order)
            out[i, -1] = p.decision.target_cash_weight
        return out

    @property
    def period_weights_array(self) -> NDArray[np.floating]:
        """Actual total weights *after* each rebalance, shape (n_periods, n_assets + 1)."""
        n = len(self.periods)
        n_assets = len(self.asset_order)
        out = np.empty((n, n_assets + 1))
        for i, p in enumerate(self.periods):
            out[i, :-1] = p.state_after.asset_weights
            out[i, -1] = float(p.state_after.cash_weight)
        return out

    @property
    def period_cash(self) -> NDArray[np.floating]:
        """Cash balance after each rebalance."""
        return np.array([p.state_after.cash for p in self.periods], dtype=float)


def _to_dates(dates: Sequence) -> list[float]:
    return as_mpl_dates(dates)


def _mark_rebalances(ax, dates: Sequence, *, label: bool = True) -> None:
    first = True
    for date in _to_dates(dates):
        ax.axvline(
            date,
            color="red",
            alpha=0.18,
            linewidth=0.8,
            label="Rebalance" if label and first else None,
            zorder=0,
        )
        first = False


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
    if len(results) == 1:
        _mark_rebalances(ax, results[0].period_execution_bars)

    format_date_axis(ax)
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
    format_date_axis(ax)
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
    w = result.period_weights_array
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
    format_date_axis(ax)
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
    format_date_axis(ax)
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
    format_date_axis(ax1)
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
    format_date_axis(ax)
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
    format_date_axis(ax)
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
    format_date_axis(ax)
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
