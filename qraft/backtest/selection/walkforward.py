from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.backtest.execution import BacktestResult
from qraft.backtest.inputs import PolicyInputsProvider, PrecomputedInputsProvider
from qraft.backtest.market import MarketData
from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.schedule import RebalanceSchedule
from qraft.backtest.selection.candidates import expand_candidates
from qraft.backtest.selection.evaluate import _shared_warmup, evaluate_candidates
from qraft.backtest.selection.results import (
    CandidateResult,
    PolicyParams,
    SelectionReport,
)
from qraft.backtest.selection.select import _METRIC, select_candidate
from qraft.backtest.simulator import precompute_inputs
from qraft.construction.policies import PolicyProtocol
from qraft.core import metrics


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward split, as inclusive ``(start, end)`` rebalance dates."""

    train: tuple[datetime, datetime]
    test: tuple[datetime, datetime]


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold: Fold
    selection: SelectionReport
    oos_summary: PerformanceSummary | None


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    folds: tuple[FoldResult, ...]
    oos_summary: PerformanceSummary | None
    oos_nav_dates: list[datetime]
    oos_nav: NDArray[np.floating]

    def folds_df(self) -> pl.DataFrame:
        """One row per fold with selected params and train/test metrics."""
        rows: list[dict[str, Any]] = []
        for i, fold_result in enumerate(self.folds):
            selected = fold_result.selection.selected
            row: dict[str, Any] = {
                "fold": i,
                "train_start": fold_result.fold.train[0],
                "train_end": fold_result.fold.train[1],
                "test_start": fold_result.fold.test[0],
                "test_end": fold_result.fold.test[1],
                "selected_params": str(fold_result.selection.selected_params or ""),
            }
            if selected is not None and selected.summary is not None:
                row.update(
                    {
                        f"train_{key}": value
                        for key, value in selected.summary.to_dict().items()
                    }
                )
            if fold_result.oos_summary is not None:
                row.update(
                    {
                        f"test_{key}": value
                        for key, value in fold_result.oos_summary.to_dict().items()
                    }
                )
            rows.append(row)
        return pl.DataFrame(rows)

    def summary_df(self) -> pl.DataFrame:
        """Single-row headline OOS summary for the stitched walk-forward track."""
        if self.oos_summary is None:
            return pl.DataFrame()
        return pl.DataFrame([self.oos_summary.to_dict()])

    def selected_params_df(self) -> pl.DataFrame:
        """Fold-by-fold selected parameter values in wide form."""
        rows: list[dict[str, Any]] = []
        for i, fold_result in enumerate(self.folds):
            params = fold_result.selection.selected_params
            row: dict[str, Any] = {
                "fold": i,
                "test_start": fold_result.fold.test[0],
                "test_end": fold_result.fold.test[1],
            }
            if params is not None:
                row.update(params.as_dict())
            rows.append(row)
        return pl.DataFrame(rows)

    def plot(self):
        """Plot a compact walk-forward dashboard. Requires matplotlib."""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        dates = self.oos_nav_dates
        nav = np.asarray(self.oos_nav, dtype=float)

        ax = axes[0, 0]
        if nav.size:
            ax.plot(dates, nav, color="steelblue", linewidth=1.6)
            for fold_result in self.folds:
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
        fold_ids = np.arange(len(self.folds))
        test_returns = [
            fr.oos_summary.total_return if fr.oos_summary is not None else np.nan
            for fr in self.folds
        ]
        colors = ["forestgreen" if r >= 0 else "crimson" for r in test_returns]
        ax.bar(fold_ids, test_returns, color=colors, alpha=0.75)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("OOS Return By Fold")
        ax.set_xlabel("Fold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1, 1]
        train_sharpe = []
        test_sharpe = []
        for fr in self.folds:
            selected = fr.selection.selected
            train_sharpe.append(
                selected.summary.sharpe
                if selected is not None and selected.summary is not None
                else np.nan
            )
            test_sharpe.append(
                fr.oos_summary.sharpe if fr.oos_summary is not None else np.nan
            )
        ax.plot(fold_ids, train_sharpe, marker="o", label="Train selected")
        ax.plot(fold_ids, test_sharpe, marker="o", label="OOS")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Sharpe By Fold")
        ax.set_xlabel("Fold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        for ax in axes.flat[:2]:
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(mdates.AutoDateLocator())
            )

        title = "Walk-Forward OOS Report"
        if self.oos_summary is not None:
            title += (
                f" | return {self.oos_summary.total_return:.1%}, "
                f"Sharpe {self.oos_summary.sharpe:.2f}, "
                f"max DD {self.oos_summary.max_drawdown:.1%}"
            )
        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()
        return fig


def walk_forward(
    dates: Sequence[datetime],
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    anchored: bool = False,
) -> list[Fold]:
    """Walk-forward folds over an ordered list of rebalance ``dates``."""
    if train_size < 1 or test_size < 1 or embargo < 0:
        raise ValueError("train_size, test_size must be >= 1 and embargo >= 0")
    advance = test_size if step is None else step
    if advance < 1:
        raise ValueError("step must be >= 1")

    ordered = list(dates)
    n = len(ordered)
    folds: list[Fold] = []
    start = 0
    while True:
        train_hi = start + train_size - 1
        test_lo = train_hi + 1 + embargo
        test_hi = test_lo + test_size - 1
        if test_hi >= n:
            break
        train_lo = 0 if anchored else start
        folds.append(
            Fold(
                train=(ordered[train_lo], ordered[train_hi]),
                test=(ordered[test_lo], ordered[test_hi]),
            )
        )
        start += advance
    return folds


def run_walk_forward(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[Any]],
    provider: PolicyInputsProvider,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    anchored: bool = False,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    step_size: int = 1,
    initial_cash: float = 100.0,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
    metric: _METRIC = "sharpe",
    max_held_fraction: float = 0.5,
) -> WalkForwardReport:
    """Walk-forward gamma-selection: run candidates once, then per fold select on
    the train window and score the choice on the held-out test window.

    Candidates are backtested ONCE over the full calendar; folds reuse those
    causal results by slicing (no re-running). The stitched OOS curve is the
    realised, look-ahead-free track record of the rolling selection.
    """
    candidates = expand_candidates(base_policy, grid)
    warmup = _shared_warmup(candidates)
    table = precompute_inputs(market, schedule, provider, warmup, step_size=step_size)
    shared = PrecomputedInputsProvider(table)
    full = evaluate_candidates(
        candidates,
        market,
        shared,
        schedule=schedule,
        step_size=step_size,
        initial_cash=initial_cash,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    folds = walk_forward(
        sorted(table),
        train_size=train_size,
        test_size=test_size,
        step=step,
        embargo=embargo,
        anchored=anchored,
    )

    fold_results: list[FoldResult] = []
    oos_returns: list[NDArray[np.floating]] = []
    oos_dates: list[datetime] = []
    for fold in folds:
        train_scores = [
            _windowed_result(cr, fold.train, periods_per_year, risk_free_rate)
            for cr in full
        ]
        selection = select_candidate(
            train_scores, metric=metric, max_held_fraction=max_held_fraction
        )
        oos_summary: PerformanceSummary | None = None
        if selection.selected_params is not None:
            winner = _find(full, selection.selected_params)
            assert winner.backtest is not None
            test_result = winner.backtest.window(fold.test[0], fold.test[1])
            oos_summary = PerformanceSummary.from_backtest(
                test_result,
                active_only=False,
                periods_per_year=periods_per_year,
                risk_free_rate=risk_free_rate,
            )
            rets = metrics.returns_from_nav(test_result.nav)
            if rets.size:
                oos_returns.append(rets)
                oos_dates.extend(test_result.nav_dates[1:])
        fold_results.append(FoldResult(fold, selection, oos_summary))

    nav, nav_dates, summary = _stitch(
        oos_returns, oos_dates, folds, initial_cash, periods_per_year, risk_free_rate
    )
    return WalkForwardReport(
        folds=tuple(fold_results),
        oos_summary=summary,
        oos_nav_dates=nav_dates,
        oos_nav=nav,
    )


def _windowed_result(
    result: CandidateResult,
    window: tuple[datetime, datetime],
    periods_per_year: float,
    risk_free_rate: float,
) -> CandidateResult:
    """Re-score a full-run candidate over a sub-window (a leakage-free slice)."""
    if result.failure is not None or result.backtest is None:
        return CandidateResult(params=result.params, failure=result.failure)
    windowed = result.backtest.window(window[0], window[1])
    summary = PerformanceSummary.from_backtest(
        windowed,
        active_only=False,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    return CandidateResult(params=result.params, summary=summary, backtest=windowed)


def _find(results: Sequence[CandidateResult], params: PolicyParams) -> CandidateResult:
    return next(r for r in results if r.params == params)


def _stitch(
    returns: Sequence[NDArray[np.floating]],
    dates: list[datetime],
    folds: Sequence[Fold],
    initial_cash: float,
    periods_per_year: float,
    risk_free_rate: float,
) -> tuple[NDArray[np.floating], list[datetime], PerformanceSummary | None]:
    """Compound the per-fold OOS return segments into one continuous curve.

    Returns are stacked (not NAV levels) because each segment comes from a
    different selected policy's run; the trading metrics on the stitched summary
    are not meaningful (read them per fold) -- the NAV metrics are the headline.
    """
    if not returns:
        return np.array([], dtype=float), [], None
    stacked = np.concatenate(returns)
    nav = np.concatenate([[initial_cash], initial_cash * np.cumprod(1.0 + stacked)])
    nav_dates = [folds[0].test[0], *dates]
    stitched = BacktestResult(
        policy_name="walk_forward_oos",
        asset_order=[],
        nav_dates=nav_dates,
        nav=nav,
        periods=[],
    )
    summary = PerformanceSummary.from_backtest(
        stitched,
        active_only=False,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    return nav, nav_dates, summary
