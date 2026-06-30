from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from qraft.backtest.execution import BacktestPeriod
from qraft.backtest.execution import BacktestResult
from qraft.backtest.metrics import PerformanceSummary
from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.core import metrics


def _result(nav: list[float], periods: list[object] | None = None) -> BacktestResult:
    return BacktestResult(
        policy_name="test",
        asset_order=[],
        nav_dates=[datetime(2024, 1, i + 1) for i in range(len(nav))],
        nav=np.array(nav, dtype=float),
        periods=periods or [],
        periods_per_year=252.0,
    )


def test_core_metrics_known_sharpe_fixture() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.03], dtype=float)

    per_period = returns.mean() / returns.std(ddof=1)
    expected = per_period * np.sqrt(252.0)

    assert metrics.per_period_sharpe(returns) == pytest.approx(per_period)
    assert metrics.sharpe(returns, 0.0, 252.0) == pytest.approx(expected)


def test_core_metrics_per_period_sharpe_handles_degenerate_series() -> None:
    assert metrics.per_period_sharpe(np.array([0.01], dtype=float)) == 0.0
    assert metrics.per_period_sharpe(np.array([0.01, 0.01], dtype=float)) == 0.0


def test_performance_summary_uses_active_window_slice() -> None:
    dates = [datetime(2024, 1, i + 1) for i in range(4)]
    state = PortfolioState(
        asset_order=["A"],
        initial_prices=np.array([1.0]),
        shares=np.array([0.0]),
        cash=100.0,
    )
    period = BacktestPeriod(
        decision_bar=dates[1],
        execution_bar=dates[2],
        state_before=state,
        decision=PolicyDecision(["A"], np.array([0.0]), 1.0),
        executed_share_trades=np.array([0.0]),
        state_after=state,
        solver_status=None,
    )
    result = BacktestResult(
        policy_name="test",
        asset_order=["A"],
        nav_dates=dates,
        nav=np.array([100.0, 50.0, 100.0, 110.0], dtype=float),
        periods=[period],
    )

    summary = PerformanceSummary.from_backtest(
        result, periods_per_year=252.0, active_only=True
    )

    assert summary.total_return == pytest.approx(0.10)
    assert summary.hit_rate == pytest.approx(1.0)


def test_cvar_empty_series_returns_nan_without_raising() -> None:
    value = metrics.cvar(np.array([], dtype=float), prob=None, distribution_type="pnl")

    assert np.isnan(value)


@pytest.mark.parametrize("nav", [[100.0], [100.0, 100.0]])
def test_performance_summary_degenerate_nav_does_not_raise_or_emit_nan(
    nav: list[float],
) -> None:
    summary = PerformanceSummary.from_backtest(_result(nav))

    values = summary.to_dict()

    assert values["annualised_vol"] == 0.0
    assert values["sharpe"] == 0.0
    assert np.isfinite(values["cvar"])
