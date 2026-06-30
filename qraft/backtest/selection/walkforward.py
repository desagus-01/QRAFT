from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.backtest.execution import BacktestResult
from qraft.backtest.inputs import PolicyInputsProvider
from qraft.backtest.market import MarketData
from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.selection.diagnostics import (
    compute_deflated_sharpe,
    compute_pbo,
    resolve_selection_periods_per_year,
    trial_sharpes,
)
from qraft.backtest.selection.evaluate import evaluate_candidate_grid
from qraft.backtest.selection.results import (
    CandidateResult,
    SelectionReport,
)
from qraft.backtest.selection.scoring import find_candidate, score_candidate_range
from qraft.backtest.selection.select import Scorer, select_candidate
from qraft.backtest.selection.splits import Fold, walk_forward_folds
from qraft.construction.optimization.moments import PolicyInputConfig
from qraft.construction.policies import PolicyProtocol
from qraft.core.configs import DEFAULT_SIMULATION_CONFIG, SimulationForecastConfig
from qraft.core import metrics
from qraft.backtest.configs import BacktestConfig, WalkForwardConfig
from qraft.forecast.run import ForecastRecipeHistory
from qraft.utils.backtest_viz import plot_walk_forward_report


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
    n_trials: int = 0
    deflated_sharpe: float | None = None
    pbo: float | None = None

    @property
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
                "selection_rule": fold_result.selection.rule,
                "n_candidates": len(fold_result.selection.candidates),
                "n_successful_candidates": sum(
                    c.summary is not None for c in fold_result.selection.candidates
                ),
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
            if selected is not None and selected.summary and fold_result.oos_summary:
                row.update(
                    {
                        "sharpe_decay": fold_result.oos_summary.sharpe
                        - selected.summary.sharpe,
                        "return_decay": fold_result.oos_summary.annualised_return
                        - selected.summary.annualised_return,
                        "vol_ratio": _safe_div(
                            fold_result.oos_summary.annualised_vol,
                            selected.summary.annualised_vol,
                        ),
                    }
                )
            rows.append(row)
        return pl.DataFrame(rows)

    @property
    def summary_df(self) -> pl.DataFrame:
        """Single-row walk-forward summary with OOS and selection diagnostics."""
        row: dict[str, Any] = {
            "n_folds": len(self.folds),
            "n_trials": self.n_trials,
            "deflated_sharpe": self.deflated_sharpe,
            "pbo": self.pbo,
        }
        if self.oos_summary is not None:
            row.update({f"oos_{k}": v for k, v in self.oos_summary.to_dict().items()})

        fold_rows = self.folds_df
        if not fold_rows.is_empty():
            for col in (
                "train_sharpe",
                "test_sharpe",
                "sharpe_decay",
                "test_total_return",
                "test_max_drawdown",
                "test_hit_rate",
                "test_held_fraction",
            ):
                if col in fold_rows.columns:
                    values = fold_rows[col].drop_nulls()
                    if not values.is_empty():
                        row[f"avg_{col}"] = values.mean()
                        row[f"median_{col}"] = values.median()
            if "selected_params" in fold_rows.columns:
                selected = [p for p in fold_rows["selected_params"].to_list() if p]
                unique_selected = set(selected)
                row["n_unique_selected"] = len(unique_selected)
                row["selection_concentration"] = _selection_concentration(selected)
                row["most_selected_params"] = _most_common(selected)
        return pl.DataFrame([row])

    @property
    def selected_params_df(self) -> pl.DataFrame:
        """Fold-by-fold selected parameter values and OOS validation metrics."""
        rows: list[dict[str, Any]] = []
        for i, fold_result in enumerate(self.folds):
            params = fold_result.selection.selected_params
            row: dict[str, Any] = {
                "fold": i,
                "test_start": fold_result.fold.test[0],
                "test_end": fold_result.fold.test[1],
                "selected_params": str(params or ""),
            }
            if params is not None:
                row.update(params.as_dict())
            if fold_result.oos_summary is not None:
                row.update(
                    {
                        "test_total_return": fold_result.oos_summary.total_return,
                        "test_sharpe": fold_result.oos_summary.sharpe,
                        "test_max_drawdown": fold_result.oos_summary.max_drawdown,
                    }
                )
            rows.append(row)
        return pl.DataFrame(rows)

    @property
    def diagnostics_df(self) -> pl.DataFrame:
        """Compact diagnostics useful for judging robustness, not just return."""
        rows = [
            ("folds", len(self.folds)),
            ("trials", self.n_trials),
            ("deflated_sharpe", self.deflated_sharpe),
            ("pbo", self.pbo),
        ]
        if self.oos_summary is not None:
            rows.extend(
                [
                    ("oos_total_return", self.oos_summary.total_return),
                    ("oos_sharpe", self.oos_summary.sharpe),
                    ("oos_max_drawdown", self.oos_summary.max_drawdown),
                    ("oos_hit_rate", self.oos_summary.hit_rate),
                ]
            )
        summary = self.summary_df
        if not summary.is_empty():
            row = summary.row(0, named=True)
            for key in (
                "avg_sharpe_decay",
                "median_sharpe_decay",
                "selection_concentration",
                "n_unique_selected",
                "most_selected_params",
            ):
                if key in row:
                    rows.append((key, row[key]))
        return pl.DataFrame(rows, schema=["metric", "value"], orient="row")

    @property
    def selection_counts_df(self) -> pl.DataFrame:
        """Frequency table for selected parameter sets across folds."""
        selected = [
            str(fr.selection.selected_params or "")
            for fr in self.folds
            if fr.selection.selected_params is not None
        ]
        counts = {params: selected.count(params) for params in set(selected)}
        rows = [
            {
                "selected_params": params,
                "n_folds": count,
                "frequency": count / len(selected),
            }
            for params, count in sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            )
        ]
        return pl.DataFrame(rows)

    def plot(self):
        """Plot a walk-forward research dashboard. Requires matplotlib."""
        return plot_walk_forward_report(
            folds=self.folds,
            folds_df=self.folds_df,
            selection_counts_df=self.selection_counts_df,
            oos_summary=self.oos_summary,
            oos_nav_dates=self.oos_nav_dates,
            oos_nav=self.oos_nav,
            n_trials=self.n_trials,
            deflated_sharpe=self.deflated_sharpe,
            pbo=self.pbo,
        )


def walk_forward(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[Any]],
    *,
    provider: PolicyInputsProvider | None = None,
    recipe_history: ForecastRecipeHistory | None = None,
    input_config: PolicyInputConfig | None = None,
    simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig = BacktestConfig(),
    score: Scorer | None = None,
) -> WalkForwardReport:
    """Walk-forward gamma-selection: run candidates once, then per fold select on
    the train window and score the choice on the held-out test window.

    Candidates are backtested ONCE over the full calendar; folds reuse those
    causal results by slicing (no re-running). The stitched OOS curve is the
    realised, look-ahead-free track record of the rolling selection.
    """
    full, dates = evaluate_candidate_grid(
        market,
        base_policy,
        grid,
        backtest_config,
        walk_config.risk_free_rate,
        provider=provider,
        recipe_history=recipe_history,
        input_config=input_config,
        simulation_config=simulation_config,
    )

    folds = walk_forward_folds(
        dates,
        train_size=walk_config.train_size,
        test_size=walk_config.test_size,
        step=walk_config.fold_step,
        embargo=walk_config.embargo,
        anchored=walk_config.anchored,
    )
    fold_results, oos_returns, oos_dates = _run_folds(
        full, folds, walk_config, backtest_config, score
    )

    periods_per_year = resolve_selection_periods_per_year(
        full, backtest_config.periods_per_year
    )
    resolved_backtest_config = BacktestConfig(
        schedule=backtest_config.schedule,
        initial_cash=backtest_config.initial_cash,
        periods_per_year=periods_per_year,
    )

    nav, nav_dates, summary = _stitch(
        oos_returns,
        oos_dates,
        folds,
        resolved_backtest_config.initial_cash,
        periods_per_year,
        walk_config.risk_free_rate,
    )
    n_trials, dsr, pbo_value = _compute_walk_forward_diagnostics(
        full, oos_returns, walk_config, resolved_backtest_config
    )

    return WalkForwardReport(
        folds=tuple(fold_results),
        oos_summary=summary,
        oos_nav_dates=nav_dates,
        oos_nav=nav,
        n_trials=n_trials,
        deflated_sharpe=dsr,
        pbo=pbo_value,
    )


def _select_fold_candidate(
    full: Sequence[CandidateResult],
    fold: Fold,
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
    score: Scorer | None,
) -> SelectionReport:
    train_scores = [
        score_candidate_range(
            cr,
            fold.train,
            backtest_config,
            walk_config.risk_free_rate,
        )
        for cr in full
    ]
    return select_candidate(
        train_scores,
        metric=walk_config.metric,
        max_held_fraction=walk_config.max_held_fraction,
        score=score,
    )


def _score_oos_fold(
    full: Sequence[CandidateResult],
    fold: Fold,
    selection: SelectionReport,
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
) -> tuple[PerformanceSummary | None, NDArray[np.floating] | None, list[datetime]]:
    if selection.selected_params is None:
        return None, None, []
    winner = find_candidate(full, selection.selected_params)
    assert winner.backtest is not None
    test_result = winner.backtest.window(fold.test[0], fold.test[1])
    oos_summary = PerformanceSummary.from_backtest(
        test_result,
        active_only=False,
        periods_per_year=backtest_config.periods_per_year,
        risk_free_rate=walk_config.risk_free_rate,
    )
    rets = metrics.returns_from_nav(test_result.nav)
    if not rets.size:
        return oos_summary, None, []
    return oos_summary, rets, list(test_result.nav_dates[1:])


def _run_folds(
    full: Sequence[CandidateResult],
    folds: Sequence[Fold],
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
    score: Scorer | None,
) -> tuple[list[FoldResult], list[NDArray[np.floating]], list[datetime]]:
    fold_results: list[FoldResult] = []
    oos_returns: list[NDArray[np.floating]] = []
    oos_dates: list[datetime] = []
    for fold in folds:
        selection = _select_fold_candidate(
            full, fold, walk_config, backtest_config, score
        )
        oos_summary, rets, dates = _score_oos_fold(
            full, fold, selection, walk_config, backtest_config
        )
        if rets is not None:
            oos_returns.append(rets)
            oos_dates.extend(dates)
        fold_results.append(FoldResult(fold, selection, oos_summary))
    return fold_results, oos_returns, oos_dates


def _compute_walk_forward_diagnostics(
    full: Sequence[CandidateResult],
    oos_returns: Sequence[NDArray[np.floating]],
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
) -> tuple[int, float | None, float | None]:
    trial_sharpe_values = trial_sharpes(full, backtest_config.periods_per_year)
    n_trials = len(trial_sharpe_values)
    dsr: float | None = None
    pbo_value: float | None = None
    if oos_returns and n_trials >= 2:
        dsr = compute_deflated_sharpe(np.concatenate(oos_returns), trial_sharpe_values)
        pbo_value = compute_pbo(full, walk_config.pbo_blocks)
    return n_trials, dsr, pbo_value


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _column_or_nan(df: pl.DataFrame, column: str, n: int) -> list[float]:
    if df.is_empty() or column not in df.columns:
        return [float("nan")] * n
    return [float(v) if v is not None else float("nan") for v in df[column].to_list()]


def _selection_concentration(selected: Sequence[str]) -> float:
    if not selected:
        return 0.0
    counts = [selected.count(params) for params in set(selected)]
    return max(counts) / len(selected)


def _most_common(selected: Sequence[str]) -> str:
    if not selected:
        return ""
    return max(set(selected), key=selected.count)


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
