from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from qraft.backtest.execution import BacktestResult
from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.selection import (
    CombinatorialCVConfig,
    Fold,
    FoldResult,
    WalkForwardConfig,
    WalkForwardReport,
    walk_forward_folds,
)
from qraft.backtest.selection.diagnostics import (
    candidate_block_returns,
    trial_sharpes,
)
from qraft.backtest.selection.results import (
    CandidateResult,
    PolicyParams,
    SelectionReport,
)
from qraft.backtest.selection.select import select_candidate

_block_returns = candidate_block_returns

DATES = [datetime(2024, 1, 1 + i) for i in range(10)]


def test_walk_forward_rolling() -> None:
    folds = walk_forward_folds(DATES, train_size=3, test_size=2)
    assert [(f.train, f.test) for f in folds] == [
        ((DATES[0], DATES[2]), (DATES[3], DATES[4])),
        ((DATES[2], DATES[4]), (DATES[5], DATES[6])),
        ((DATES[4], DATES[6]), (DATES[7], DATES[8])),
    ]


def test_walk_forward_anchored_expands_train() -> None:
    folds = walk_forward_folds(DATES, train_size=3, test_size=2, anchored=True)
    assert [f.train[0] for f in folds] == [DATES[0], DATES[0], DATES[0]]
    assert [f.train[1] for f in folds] == [DATES[2], DATES[4], DATES[6]]


def test_walk_forward_embargo_shifts_test() -> None:
    folds = walk_forward_folds(DATES, train_size=3, test_size=2, embargo=1)
    # train [0,2], embargo skips d3, test [4,5]
    assert folds[0].train == (DATES[0], DATES[2])
    assert folds[0].test == (DATES[4], DATES[5])


def test_walk_forward_custom_step_overlaps() -> None:
    folds = walk_forward_folds(DATES, train_size=3, test_size=2, step=1)
    assert folds[0].test == (DATES[3], DATES[4])
    assert folds[1].test == (DATES[4], DATES[5])  # step=1 -> overlapping tests


def test_walk_forward_too_short_returns_empty() -> None:
    assert walk_forward_folds(DATES[:3], train_size=3, test_size=2) == []


def test_walk_forward_validates() -> None:
    with pytest.raises(ValueError):
        walk_forward_folds(DATES, train_size=0, test_size=2)


def test_walk_forward_config_validates() -> None:
    with pytest.raises(ValueError, match="train_size"):
        WalkForwardConfig(train_size=0, test_size=2)

    with pytest.raises(ValueError, match="test_size must be >= 2"):
        WalkForwardConfig(train_size=3, test_size=1)


def test_walk_forward_config_constructs_with_inherited_slots() -> None:
    config = WalkForwardConfig(risk_free_rate=0.01, train_size=3, test_size=2)

    assert config.risk_free_rate == 0.01
    assert config.train_size == 3
    assert config.test_size == 2


def test_combinatorial_cv_config_defaults_are_conservative() -> None:
    config = CombinatorialCVConfig()

    assert config.n_groups == 10
    assert config.n_test_groups == 3
    assert config.purge == 1
    assert config.embargo == 1


def test_walk_forward_folds_accepts_config_values() -> None:
    config = WalkForwardConfig(train_size=3, test_size=2, anchored=True)

    assert walk_forward_folds(
        DATES,
        train_size=config.train_size,
        test_size=config.test_size,
        anchored=config.anchored,
    ) == walk_forward_folds(
        DATES,
        train_size=3,
        test_size=2,
        anchored=True,
    )


def test_walk_forward_report_helpers() -> None:
    params = PolicyParams.of(risk_aversion=2)
    summary = PerformanceSummary(
        total_return=0.10,
        annualised_return=0.20,
        annualised_vol=0.15,
        sharpe=1.30,
        sortino=1.50,
        max_drawdown=-0.05,
        calmar=4.0,
        cvar=-0.02,
        hit_rate=0.55,
        avg_turnover=0.10,
        total_cost=0.01,
        n_periods=2,
        n_warnings=0,
        n_solver_failures=0,
        held_fraction=0.0,
    )
    selection = SelectionReport(
        candidates=(CandidateResult(params=params, summary=summary),),
        selected_params=params,
        rule="max_sharpe",
    )
    report = WalkForwardReport(
        folds=(
            FoldResult(
                fold=Fold(train=(DATES[0], DATES[2]), test=(DATES[3], DATES[4])),
                selection=selection,
                oos_summary=summary,
            ),
        ),
        oos_summary=summary,
        oos_nav_dates=[DATES[3], DATES[4]],
        oos_nav=np.array([100.0, 110.0]),
    )

    folds_df = report.folds_df
    assert folds_df["selected_params"].to_list() == ["risk_aversion=2"]
    assert folds_df["train_sharpe"].to_list() == [1.30]
    assert folds_df["test_total_return"].to_list() == [0.10]
    assert report.summary_df["oos_sharpe"].to_list() == [1.30]
    assert report.selected_params_df["risk_aversion"].to_list() == [2]
    assert report.diagnostics_df["metric"].to_list()[0] == "folds"
    assert report.selection_counts_df["n_folds"].to_list() == [1]


def test_block_returns_validates_minimum_blocks() -> None:
    with pytest.raises(ValueError, match="n_blocks must be >= 2"):
        _block_returns([], 1)


def test_block_returns_requires_successful_candidates() -> None:
    with pytest.raises(ValueError, match="at least 2 successful"):
        _block_returns([_candidate_result(1, DATES[:4])], 2)


def test_block_returns_diagnoses_candidate_date_mismatch() -> None:
    shifted = [*DATES[:2], datetime(2024, 2, 1), DATES[3]]

    with pytest.raises(ValueError, match="NAV dates do not match.*index 2"):
        _block_returns(
            [_candidate_result(1, DATES[:4]), _candidate_result(2, shifted)], 2
        )


def test_block_returns_diagnoses_nav_date_length_mismatch() -> None:
    result = _candidate_result(1, DATES[:4])
    assert result.backtest is not None
    broken = CandidateResult(
        params=result.params,
        backtest=BacktestResult(
            policy_name="broken",
            asset_order=[],
            nav_dates=result.backtest.nav_dates[:-1],
            nav=result.backtest.nav,
            periods=[],
        ),
    )

    with pytest.raises(ValueError, match="nav_dates but .* NAV values"):
        _block_returns([broken, _candidate_result(2, DATES[:4])], 2)


def test_block_returns_builds_block_candidate_matrix() -> None:
    perf = _block_returns(
        [_candidate_result(1, DATES[:4]), _candidate_result(2, DATES[:4])], 2
    )

    assert perf.shape == (2, 2)


def test_trial_sharpes_infers_periods_per_year_from_candidate_dates() -> None:
    summary = PerformanceSummary(
        total_return=0.10,
        annualised_return=0.20,
        annualised_vol=0.15,
        sharpe=2.0,
        sortino=2.0,
        max_drawdown=-0.05,
        calmar=4.0,
        cvar=-0.02,
        hit_rate=0.55,
        avg_turnover=0.10,
        total_cost=0.01,
        n_periods=2,
        n_warnings=0,
        n_solver_failures=0,
        held_fraction=0.0,
    )
    candidate = _candidate_result(1, DATES[:4])
    candidate = CandidateResult(
        params=candidate.params,
        summary=summary,
        backtest=candidate.backtest,
    )

    values = trial_sharpes([candidate], None)

    assert values == pytest.approx([2.0 / np.sqrt(252.0)])


def test_selection_excludes_nan_scores() -> None:
    flat = CandidateResult(
        params=PolicyParams.of(risk_aversion=0.0),
        summary=PerformanceSummary.from_backtest(
            BacktestResult(
                policy_name="flat",
                asset_order=[],
                nav_dates=DATES[:3],
                nav=np.array([100.0, 100.0, 100.0]),
                periods=[],
                periods_per_year=252.0,
            )
        ),
    )
    losing = CandidateResult(
        params=PolicyParams.of(risk_aversion=1.0),
        summary=PerformanceSummary.from_backtest(
            BacktestResult(
                policy_name="losing",
                asset_order=[],
                nav_dates=DATES[:4],
                nav=np.array([100.0, 99.0, 98.5, 98.0]),
                periods=[],
                periods_per_year=252.0,
            )
        ),
    )

    report = select_candidate([flat, losing], metric="sharpe")

    assert report.selected_params == losing.params


def _candidate_result(value: int, dates: list[datetime]) -> CandidateResult:
    nav = np.arange(100.0 + value, 100.0 + value + len(dates), dtype=float)
    return CandidateResult(
        params=PolicyParams.of(value=value),
        backtest=BacktestResult(
            policy_name=f"candidate_{value}",
            asset_order=[],
            nav_dates=dates,
            nav=nav,
            periods=[],
        ),
    )
