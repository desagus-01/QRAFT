from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.backtest.configs import BacktestConfig, CombinatorialCVConfig
from qraft.core.market import MarketData
from qraft.backtest.result import PerformanceSummary
from qraft.backtest.selection.diagnostics import (
    compute_deflated_sharpe,
    compute_pbo,
    resolve_selection_periods_per_year,
    trial_sharpes,
)
from qraft.backtest.selection.evaluate import (
    SelectionInputSource,
    evaluate_candidate_grid,
)
from qraft.backtest.selection.evaluation import CandidateEvaluation
from qraft.backtest.selection.results import CandidateResult, PolicyParams
from qraft.backtest.selection.scoring import (
    ScoreSpec,
    find_candidate,
    returns_for_range,
    score_candidate_ranges,
    summary_from_returns,
)
from qraft.backtest.selection.select import select_candidate
from qraft.backtest.selection.splits import (
    CombinatorialFold,
    combinatorial_purged_folds,
)
from qraft.construction.optimization.inputs import InputPlan
from qraft.construction.policies import PolicyProtocol
from qraft.forecast.forecaster import Forecaster
from qraft.utils.log import info_event
from qraft.backtest.selection.render import plot_combinatorial_report


T = TypeVar("T")
logger = logging.getLogger(__name__)


def _assign_paths(
    n_groups: int,
    n_test_groups: int,
    fold_test_group_values: Sequence[Mapping[int, T | None]],
) -> list[list[T | None]]:
    """Recombine fold-level held-out group returns into complete CPCV paths.

    Each date group appears in ``C(n_groups - 1, n_test_groups - 1)`` folds.
    CPCV turns those repeated held-out evaluations into that many OOS paths by
    placing the first evaluation for every group in path 0, the second in path 1,
    and so on. The inner list stays group-indexed so returns can be concatenated
    in chronological group order.
    """
    n_paths = math.comb(n_groups - 1, n_test_groups - 1)
    path_segments_by_group: list[list[T | None]] = [
        [None] * n_groups for _ in range(n_paths)
    ]
    next_path_by_group = [0] * n_groups
    for test_group_values in fold_test_group_values:
        for group_id, value in test_group_values.items():
            path_index = next_path_by_group[group_id]
            if path_index < n_paths:
                path_segments_by_group[path_index][group_id] = value
            next_path_by_group[group_id] += 1
    return path_segments_by_group


@dataclass(frozen=True, slots=True)
class CombinatorialReport:
    """Distribution of OOS paths from CPCV, plus overfitting diagnostics."""

    paths: tuple[PerformanceSummary, ...]
    path_sharpes: NDArray[np.floating]
    n_groups: int
    n_test_groups: int
    purge: int
    embargo: int
    n_trials: int
    evaluation: CandidateEvaluation | None = None
    folds: tuple[CombinatorialFold, ...] = ()
    path_returns: tuple[NDArray[np.floating], ...] = ()
    selected_params: PolicyParams | None = None
    fold_selected_params: tuple[PolicyParams | None, ...] = ()
    pbo: float | None = None
    deflated_sharpe: float | None = None

    def require(self, *, max_pbo: float | None = None) -> "CombinatorialReport":
        if max_pbo is not None and self.pbo is not None and self.pbo > max_pbo:
            raise ValueError(
                f"Combinatorial PBO {self.pbo:.3f} exceeds required maximum "
                f"{max_pbo:.3f}."
            )
        return self

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    @property
    def median_sharpe(self) -> float:
        return (
            float(np.median(self.path_sharpes)) if self.path_sharpes.size else math.nan
        )

    @property
    def sharpe_iqr(self) -> tuple[float, float]:
        if self.path_sharpes.size == 0:
            return math.nan, math.nan
        return (
            float(np.quantile(self.path_sharpes, 0.25)),
            float(np.quantile(self.path_sharpes, 0.75)),
        )

    @property
    def worst_sharpe(self) -> float:
        return float(self.path_sharpes.min()) if self.path_sharpes.size else math.nan

    def summary_df(self) -> pl.DataFrame:
        lo, hi = self.sharpe_iqr
        return pl.DataFrame(
            [
                {
                    "n_paths": self.n_paths,
                    "n_groups": self.n_groups,
                    "n_test_groups": self.n_test_groups,
                    "n_trials": self.n_trials,
                    "median_sharpe": self.median_sharpe,
                    "sharpe_q25": lo,
                    "sharpe_q75": hi,
                    "worst_sharpe": self.worst_sharpe,
                    "deflated_sharpe": self.deflated_sharpe,
                    "pbo": self.pbo,
                }
            ]
        )

    def plot(self):
        """Plot a CPCV research dashboard. Requires matplotlib."""
        return plot_combinatorial_report(
            paths=self.paths,
            path_sharpes=self.path_sharpes,
            path_returns=self.path_returns,
            n_paths=self.n_paths,
            n_groups=self.n_groups,
            n_test_groups=self.n_test_groups,
            n_trials=self.n_trials,
            median_sharpe=self.median_sharpe,
            sharpe_iqr=self.sharpe_iqr,
            worst_sharpe=self.worst_sharpe,
            deflated_sharpe=self.deflated_sharpe,
            pbo=self.pbo,
        )


def combinatorial_purged(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[Any]],
    *,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
    source: SelectionInputSource | None = None,
    cv_config: CombinatorialCVConfig,
    backtest_config: BacktestConfig = BacktestConfig(),
    score: ScoreSpec | None = None,
) -> CombinatorialReport:
    """CPCV gamma-selection: run candidates once, select per combination, then
    recombine held-out test groups into a *distribution* of OOS paths.

    ``cv_config`` supplies selection and CPCV settings. ``backtest_config``
    supplies execution settings.
    """
    evaluation = evaluate_candidate_grid(
        market,
        base_policy,
        grid,
        backtest_config,
        cv_config.cv_config.risk_free_rate,
        forecaster=forecaster,
        source=source,
        plan=plan,
    )
    return combinatorial_from_evaluation(
        evaluation,
        cv_config=cv_config,
        score=score,
    )


def combinatorial_from_evaluation(
    evaluation: CandidateEvaluation,
    *,
    cv_config: CombinatorialCVConfig,
    score: ScoreSpec | None = None,
) -> CombinatorialReport:
    candidate_results = evaluation.candidate_results
    backtest_config = evaluation.backtest_config
    folds = combinatorial_purged_folds(
        evaluation.dates,
        n_groups=cv_config.n_groups,
        n_test_groups=cv_config.n_test_groups,
        purge=cv_config.purge,
        embargo=cv_config.embargo,
    )
    info_event(
        logger,
        "validation.combinatorial_started",
        "Combinatorial validation started",
        folds=len(folds),
        candidates=len(candidate_results),
        groups=cv_config.n_groups,
        test_groups=cv_config.n_test_groups,
        purge=cv_config.purge,
        embargo=cv_config.embargo,
    )
    periods_per_year = resolve_selection_periods_per_year(
        candidate_results, backtest_config.periods_per_year
    )
    resolved_backtest_config = BacktestConfig(
        schedule=backtest_config.schedule,
        initial_cash=backtest_config.initial_cash,
        periods_per_year=periods_per_year,
    )
    fold_test_group_returns: list[dict[int, NDArray[np.floating] | None]] = []
    fold_selected_params: list[PolicyParams | None] = []
    for fold_index, fold in enumerate(folds):
        train_scores = [
            score_candidate_ranges(
                candidate_result,
                fold.train,
                resolved_backtest_config,
                cv_config.cv_config.risk_free_rate,
            )
            for candidate_result in candidate_results
        ]
        selection = select_candidate(
            train_scores,
            max_held_fraction=cv_config.cv_config.max_held_fraction,
            score=score if score is not None else cv_config.cv_config.metric,
        )
        fold_selected_params.append(selection.selected_params)
        winner = (
            find_candidate(candidate_results, selection.selected_params)
            if selection.selected_params is not None
            else None
        )
        test_group_returns: dict[int, NDArray[np.floating] | None] = {}
        for group_id, (start, end) in zip(fold.test_groups, fold.test, strict=True):
            if winner is not None and winner.backtest is not None:
                group_returns = returns_for_range(winner.backtest, (start, end))
                test_group_returns[group_id] = (
                    group_returns if group_returns.size else None
                )
            else:
                test_group_returns[group_id] = None
        fold_test_group_returns.append(test_group_returns)
        info_event(
            logger,
            "validation.fold_completed",
            "Combinatorial fold completed",
            fold=fold_index,
            selected_params=selection.selected_params,
            test_groups=tuple(fold.test_groups),
        )

    paths = _assign_paths(
        cv_config.n_groups, cv_config.n_test_groups, fold_test_group_returns
    )
    path_returns = [
        np.concatenate(parts)
        for path in paths
        if (parts := [s for s in path if s is not None and s.size])
    ]
    summaries = [
        summary_from_returns(
            r,
            resolved_backtest_config,
            cv_config.cv_config.risk_free_rate,
            policy_name="cpcv_path",
        )
        for r in path_returns
    ]
    valid = [(r, s) for r, s in zip(path_returns, summaries) if s is not None]
    summary_tuple = tuple(s for _, s in valid)
    sharpes = np.array([s.sharpe for s in summary_tuple], dtype=float)
    n_trials, dsr, pbo_value = _diagnostics(
        candidate_results, valid, sharpes, cv_config, resolved_backtest_config
    )
    selected_params = _most_common_params(
        [p for p in fold_selected_params if p is not None]
    )
    info_event(
        logger,
        "validation.combinatorial_completed",
        "Combinatorial validation completed",
        folds=len(folds),
        paths=len(summary_tuple),
        trials=n_trials,
        selected_params=selected_params,
        median_sharpe=(f"{float(np.median(sharpes)):.3f}" if sharpes.size else None),
        pbo=(f"{pbo_value:.3f}" if pbo_value is not None else None),
        deflated_sharpe=(f"{dsr:.3f}" if dsr is not None else None),
    )
    return CombinatorialReport(
        paths=summary_tuple,
        path_sharpes=sharpes,
        n_groups=cv_config.n_groups,
        n_test_groups=cv_config.n_test_groups,
        purge=cv_config.purge,
        embargo=cv_config.embargo,
        n_trials=n_trials,
        evaluation=evaluation,
        folds=tuple(folds),
        path_returns=tuple(r for r, _ in valid),
        selected_params=selected_params,
        fold_selected_params=tuple(fold_selected_params),
        pbo=pbo_value,
        deflated_sharpe=dsr,
    )


def _most_common_params(params: Sequence[PolicyParams]) -> PolicyParams | None:
    if not params:
        return None
    return max(set(params), key=params.count)


def _diagnostics(
    candidate_results: Sequence[CandidateResult],
    valid: Sequence[tuple[NDArray[np.floating], PerformanceSummary]],
    sharpes: NDArray[np.floating],
    cv_config: CombinatorialCVConfig,
    backtest_config: BacktestConfig,
) -> tuple[int, float | None, float | None]:
    trial_sharpe_values = trial_sharpes(
        candidate_results, backtest_config.periods_per_year
    )
    n_trials = len(trial_sharpe_values)
    dsr = None
    if valid and n_trials >= 2:
        median_returns = valid[int(np.argsort(sharpes)[len(sharpes) // 2])][0]
        dsr = compute_deflated_sharpe(median_returns, trial_sharpe_values)
    pbo_value = None
    if cv_config.n_groups % 2 == 0:
        pbo_blocks = max(cv_config.cv_config.pbo_blocks, cv_config.n_groups)
        pbo_value = compute_pbo(candidate_results, pbo_blocks)
    return n_trials, dsr, pbo_value
