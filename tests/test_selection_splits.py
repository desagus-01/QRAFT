from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.selection import Fold, FoldResult, WalkForwardReport, walk_forward
from qraft.backtest.selection.results import (
    CandidateResult,
    PolicyParams,
    SelectionReport,
)

DATES = [datetime(2024, 1, 1 + i) for i in range(10)]


def test_walk_forward_rolling() -> None:
    folds = walk_forward(DATES, train_size=3, test_size=2)
    assert [(f.train, f.test) for f in folds] == [
        ((DATES[0], DATES[2]), (DATES[3], DATES[4])),
        ((DATES[2], DATES[4]), (DATES[5], DATES[6])),
        ((DATES[4], DATES[6]), (DATES[7], DATES[8])),
    ]


def test_walk_forward_anchored_expands_train() -> None:
    folds = walk_forward(DATES, train_size=3, test_size=2, anchored=True)
    assert [f.train[0] for f in folds] == [DATES[0], DATES[0], DATES[0]]
    assert [f.train[1] for f in folds] == [DATES[2], DATES[4], DATES[6]]


def test_walk_forward_embargo_shifts_test() -> None:
    folds = walk_forward(DATES, train_size=3, test_size=2, embargo=1)
    # train [0,2], embargo skips d3, test [4,5]
    assert folds[0].train == (DATES[0], DATES[2])
    assert folds[0].test == (DATES[4], DATES[5])


def test_walk_forward_custom_step_overlaps() -> None:
    folds = walk_forward(DATES, train_size=3, test_size=2, step=1)
    assert folds[0].test == (DATES[3], DATES[4])
    assert folds[1].test == (DATES[4], DATES[5])  # step=1 -> overlapping tests


def test_walk_forward_too_short_returns_empty() -> None:
    assert walk_forward(DATES[:3], train_size=3, test_size=2) == []


def test_walk_forward_validates() -> None:
    with pytest.raises(ValueError):
        walk_forward(DATES, train_size=0, test_size=2)


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

    folds_df = report.folds_df()
    assert folds_df["selected_params"].to_list() == ["risk_aversion=2"]
    assert folds_df["train_sharpe"].to_list() == [1.30]
    assert folds_df["test_total_return"].to_list() == [0.10]
    assert report.summary_df()["sharpe"].to_list() == [1.30]
    assert report.selected_params_df()["risk_aversion"].to_list() == [2]
