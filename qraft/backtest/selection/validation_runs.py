from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from qraft.backtest.configs import (
    BacktestConfig,
    CombinatorialCVConfig,
    WalkForwardConfig,
)
from qraft.backtest.result import BacktestResult, PerformanceSummary
from qraft.backtest.selection.candidate_eval import CandidateEvaluation
from qraft.backtest.selection.diagnostics import (
    compute_deflated_sharpe,
    compute_pbo,
    resolve_selection_periods_per_year,
    trial_sharpes,
)
from qraft.backtest.selection.grid_eval import (
    SelectionInputSource,
    evaluate_candidate_grid,
)
from qraft.backtest.selection.reports import (
    CombinatorialReport,
    FoldResult,
    WalkForwardReport,
)
from qraft.backtest.selection.results import (
    CandidateResult,
    PolicyParams,
    SelectionReport,
)
from qraft.backtest.selection.scoring import (
    ScoreSpec,
    find_candidate,
    returns_for_range,
    score_candidate_range,
    score_candidate_ranges,
    summary_from_returns,
)
from qraft.backtest.selection.select import select_candidate
from qraft.backtest.selection.splits import (
    Fold,
    combinatorial_purged_folds,
    walk_forward_folds,
)
from qraft.construction.optimization.inputs import InputPlan
from qraft.construction.policies import PolicyProtocol
from qraft.core import metrics
from qraft.core.market import MarketData
from qraft.forecast.forecaster import Forecaster
from qraft.utils.log import info_event

logger = logging.getLogger(__name__)


def walk_forward(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[object]],
    *,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
    source: SelectionInputSource | None = None,
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig = BacktestConfig(),
    score: ScoreSpec | None = None,
) -> WalkForwardReport:
    evaluation = evaluate_candidate_grid(
        market,
        base_policy,
        grid,
        backtest_config,
        walk_config.risk_free_rate,
        forecaster=forecaster,
        source=source,
        plan=plan,
    )
    return walk_forward_from_evaluation(
        evaluation,
        walk_config=walk_config,
        score=score,
    )


def walk_forward_from_evaluation(
    evaluation: CandidateEvaluation,
    *,
    walk_config: WalkForwardConfig,
    score: ScoreSpec | None = None,
) -> WalkForwardReport:
    full = evaluation.candidate_results
    backtest_config = evaluation.backtest_config

    folds = walk_forward_folds(
        evaluation.dates,
        train_size=walk_config.train_size,
        test_size=walk_config.test_size,
        step=walk_config.fold_step,
        embargo=walk_config.embargo,
        anchored=walk_config.anchored,
    )
    info_event(
        logger,
        "validation.walk_forward_started",
        "Walk-forward validation started",
        folds=len(folds),
        candidates=len(full),
        train_size=walk_config.train_size,
        test_size=walk_config.test_size,
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

    info_event(
        logger,
        "validation.walk_forward_completed",
        "Walk-forward validation completed",
        folds=len(fold_results),
        trials=n_trials,
        oos_sharpe=(f"{summary.sharpe:.3f}" if summary is not None else None),
        pbo=(f"{pbo_value:.3f}" if pbo_value is not None else None),
        deflated_sharpe=(f"{dsr:.3f}" if dsr is not None else None),
    )
    return WalkForwardReport(
        folds=tuple(fold_results),
        oos_summary=summary,
        oos_nav_dates=nav_dates,
        oos_nav=nav,
        evaluation=evaluation,
        n_trials=n_trials,
        deflated_sharpe=dsr,
        pbo=pbo_value,
    )


def combinatorial_purged(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[object]],
    *,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
    source: SelectionInputSource | None = None,
    cv_config: CombinatorialCVConfig,
    backtest_config: BacktestConfig = BacktestConfig(),
    score: ScoreSpec | None = None,
) -> CombinatorialReport:
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


def _select_fold_candidate(
    full: Sequence[CandidateResult],
    fold: Fold,
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
    score: ScoreSpec | None,
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
        max_held_fraction=walk_config.max_held_fraction,
        score=score if score is not None else walk_config.metric,
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
    score: ScoreSpec | None,
) -> tuple[list[FoldResult], list[NDArray[np.floating]], list[datetime]]:
    fold_results: list[FoldResult] = []
    oos_returns: list[NDArray[np.floating]] = []
    oos_dates: list[datetime] = []
    for fold_index, fold in enumerate(folds):
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
        info_event(
            logger,
            "validation.fold_completed",
            "Walk-forward fold completed",
            fold=fold_index,
            selected_params=selection.selected_params,
            test_sharpe=(
                f"{oos_summary.sharpe:.3f}" if oos_summary is not None else None
            ),
            test_start=fold.test[0],
            test_end=fold.test[1],
        )
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


def _stitch(
    returns: Sequence[NDArray[np.floating]],
    dates: list[datetime],
    folds: Sequence[Fold],
    initial_cash: float,
    periods_per_year: float,
    risk_free_rate: float,
) -> tuple[NDArray[np.floating], list[datetime], PerformanceSummary | None]:
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


def _assign_paths(
    n_groups: int,
    n_test_groups: int,
    fold_test_group_values: Sequence[Mapping[int, NDArray[np.floating] | None]],
) -> list[list[NDArray[np.floating] | None]]:
    n_paths = math.comb(n_groups - 1, n_test_groups - 1)
    path_segments_by_group: list[list[NDArray[np.floating] | None]] = [
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
