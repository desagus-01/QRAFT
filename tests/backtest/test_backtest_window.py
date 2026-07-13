from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from conftest import portfolio_state
from qraft.backtest.result import BacktestPeriod, BacktestResult, PerformanceSummary
from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState

DATES = [datetime(2024, 1, d) for d in (1, 2, 3, 4, 5, 8)]


def _state() -> PortfolioState:
    return portfolio_state(["A"], [1.0], [0.0], 100.0)


def _period(decision: datetime, execution: datetime) -> BacktestPeriod:
    state = _state()
    return BacktestPeriod(
        decision_bar=decision,
        execution_bar=execution,
        state_before=state,
        decision=PolicyDecision(["A"], np.array([0.0]), 1.0),
        executed_share_trades=np.array([0.0]),
        state_after=state,
        solver_status="optimal",
    )


def _result() -> BacktestResult:
    return BacktestResult(
        policy_name="p",
        asset_order=["A"],
        nav_dates=DATES,
        nav=np.array([100.0, 101, 102, 103, 104, 105]),
        periods=[_period(DATES[0], DATES[1]), _period(DATES[3], DATES[4])],
        holding_costs=np.zeros(6),
    )


def test_window_slices_bars_and_periods() -> None:
    w = _result().window(DATES[2], DATES[4])  # [01-03, 01-05] inclusive
    assert w.nav_dates == DATES[2:5]
    np.testing.assert_allclose(w.nav, [102, 103, 104])
    assert w.holding_costs.size == 3
    # period executing at 01-02 excluded; period executing at 01-05 included
    assert [p.execution_bar for p in w.periods] == [DATES[4]]


def test_windowed_summary_matches_slice() -> None:
    w = _result().window(DATES[2], DATES[4])
    summary = PerformanceSummary.from_backtest(w, active_only=False)
    assert summary.total_return == pytest.approx(104 / 102 - 1)


def test_window_out_of_range_returns_empty_result() -> None:
    w = _result().window(datetime(2025, 1, 1), datetime(2025, 1, 2))
    assert w.nav.size == 0
    assert w.nav_dates == []
    assert w.periods == []
    assert w.holding_costs.size == 0
