from datetime import datetime

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

from conftest import portfolio_state
from qraft.backtest.costs import CostModel
from qraft.backtest.execution import execute_frictionless
from qraft.backtest.result.period import BacktestPeriod
from qraft.backtest.result.result import BacktestResult, plot_nav
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.optimization.objectives.specs import (
    ExpectedReturn,
    TransactionCost,
)
from qraft.construction.optimization.problem import MPOProblemBuilder
from qraft.construction.policies import (
    Allocation,
    EqualWeightPolicy,
    MPOPolicy,
    PolicyDecision,
    PolicyProjection,
    PolicyRun,
    run_policy,
)
from qraft.construction.state import PortfolioState
from qraft.core.market import MarketData
from qraft.core.universe import AssetUniverse
from qraft.forecast.forecast_paths import ForecastPaths


def _forecasts() -> ForecastPaths:
    return ForecastPaths(
        asset_paths={
            "A": np.array([[11.0, 12.0], [9.0, 10.0]]),
            "B": np.array([[22.0, 24.0], [18.0, 20.0]]),
        },
        dates=pl.Series("date", [datetime(2024, 1, 3), datetime(2024, 1, 4)]),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 10.0, "B": 20.0},
        universe=AssetUniverse.factors_free(["A", "B"]),
    )


def _state() -> PortfolioState:
    return portfolio_state(["A", "B"], [10.0, 20.0], [0.0, 0.0], 100.0)


def test_policy_decide_returns_decision_without_projecting() -> None:
    decision = EqualWeightPolicy(target_cash_weight=0.2).decide(_state(), _forecasts())

    assert isinstance(decision, PolicyDecision)
    assert decision.asset_order == ["A", "B"]
    np.testing.assert_allclose(decision.target_weights_risk, [0.4, 0.4])
    assert decision.target_cash_weight == 0.2


def test_equal_weight_policy_uses_policy_input_cash_return() -> None:
    inputs = OptimizerInputs.from_arrays(
        assets=["A", "B"],
        mean=np.ones((1, 2)),
        cash_return=np.array([0.001]),
    )

    decision = EqualWeightPolicy(target_cash_weight=0.2).decide(_state(), inputs)

    np.testing.assert_allclose(decision.cash_return, [0.001])


def test_portfolio_state_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="initial_prices must be strictly positive"):
        PortfolioState(["A"], np.array([0.0]), np.array([1.0]), 100.0)


def test_execute_frictionless_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="execution prices must be strictly positive"):
        execute_frictionless(
            PolicyDecision(["A"], np.array([1.0]), 0.0),
            shares=np.array([0.0]),
            cash=100.0,
            prices=np.array([0.0]),
            asset_order=["A"],
        )


def test_execute_frictionless_finances_trade_cost_inside_nav() -> None:
    execution = execute_frictionless(
        PolicyDecision(["A", "B"], np.array([0.5, 0.5]), 0.0),
        shares=np.array([0.0, 0.0]),
        cash=100.0,
        prices=np.array([10.0, 20.0]),
        asset_order=["A", "B"],
        costs=CostModel(transaction=TransactionCost(cost=0.01)),
    )

    assert execution.cash >= 0.0
    assert execution.trade_cost == pytest.approx(1.0)
    assert execution.shares @ np.array([10.0, 20.0]) + execution.cash == pytest.approx(
        99.0
    )
    np.testing.assert_allclose(
        execution.executed_shares * np.array([10.0, 20.0]), [49.5, 49.5]
    )


def test_forecast_at_step_repeats_horizon_date() -> None:
    panel = _forecasts().at_step(2)

    assert panel.dates.to_list() == [datetime(2024, 1, 4), datetime(2024, 1, 4)]


def test_projection_is_created_independently_from_decision() -> None:
    state = _state()
    forecasts = _forecasts()
    decision = EqualWeightPolicy(target_cash_weight=0.2).decide(state, forecasts)

    projection = PolicyProjection.from_decision(decision, forecasts, state)

    assert isinstance(projection, PolicyProjection)
    np.testing.assert_allclose(
        projection.forecast_values,
        [[108.0, 116.0], [92.0, 100.0]],
    )


def test_run_policy_orchestrates_decision_and_projection() -> None:
    optimizer_inputs = OptimizerInputs.from_arrays(
        assets=["A", "B"],
        mean=np.ones((1, 2)),
        cash_return=np.array([0.001]),
    )
    result = run_policy(
        EqualWeightPolicy(target_cash_weight=0.2),
        _state(),
        _forecasts(),
        optimizer_inputs=optimizer_inputs,
        as_of=datetime(2024, 1, 2),
    )

    assert isinstance(result, PolicyRun)
    assert isinstance(result.decision, PolicyDecision)
    assert isinstance(result.projection, PolicyProjection)
    assert result.forecasts is not None
    assert result.optimizer_inputs is optimizer_inputs
    assert result.as_of == datetime(2024, 1, 2)
    assert result.target_weights == {"A": 0.4, "B": 0.4}
    assert result.projection.target_weights == {"A": 0.4, "B": 0.4}
    assert result.projection.total_target_weights == {"A": 0.4, "B": 0.4, "cash": 0.2}
    assert result.projection.target_cash_weight == result.decision.target_cash_weight
    np.testing.assert_allclose(
        result.projection.target_weights_risk,
        result.decision.target_weights_risk,
    )


def test_policy_run_exposes_mpo_diagnostics() -> None:
    policy = MPOPolicy(problem=MPOProblemBuilder().add(ExpectedReturn()).build())
    optimizer_inputs = OptimizerInputs.from_arrays(
        assets=["A"],
        mean=np.array([[0.01]]),
        covariances=np.array([[[0.04]]]),
        scenario_returns=np.array([[[0.02]], [[-0.01]]]),
        scenario_probs=np.array([0.5, 0.5]),
        cash_return=np.array([0.0]),
    )

    run = run_policy(policy, _state(), _forecasts(), optimizer_inputs=optimizer_inputs)

    metrics = run.plan_metrics()
    assert set(metrics) == {"expected_return", "volatility"}
    assert isinstance(run.terminal_cvar(), float)
    assert isinstance(run.in_model_cvar(), float)


def test_policy_run_diagnostics_error_without_mpo_result() -> None:
    run = run_policy(
        EqualWeightPolicy(target_cash_weight=0.2),
        _state(),
        _forecasts(),
    )

    with pytest.raises(ValueError, match="MPOResult diagnostics"):
        run.plan_metrics()


def test_policy_run_plot_weights_returns_figure() -> None:
    run = run_policy(
        EqualWeightPolicy(target_cash_weight=0.2),
        _state(),
        _forecasts(),
    )

    fig = run.plot_weights()

    assert isinstance(fig, Figure)


def test_backtest_nav_plot_marks_rebalances() -> None:
    state_before = _state()
    state_after = _state()
    period = BacktestPeriod(
        decision_bar=datetime(2024, 1, 2),
        execution_bar=datetime(2024, 1, 3),
        state_before=state_before,
        decision=PolicyDecision(["A", "B"], np.array([0.4, 0.4]), 0.2),
        executed_share_trades=np.array([1.0, 1.0]),
        state_after=state_after,
        solver_status="ok",
    )
    result = BacktestResult(
        policy_name="test",
        asset_order=["A", "B"],
        nav_dates=[datetime(2024, 1, 2), datetime(2024, 1, 3)],
        nav=np.array([100.0, 101.0]),
        periods=[period],
    )

    fig = plot_nav(result)

    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "Rebalance" in labels


def test_allocation_returns_run_with_same_forecasts(monkeypatch) -> None:
    market = MarketData.from_prices(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "A": [10.0, 10.0],
                "B": [20.0, 20.0],
            }
        ),
        AssetUniverse.factors_free(["A", "B"]),
    )
    forecasts = _forecasts()
    captured = {}

    def fake_build_optimizer_input_table(snapshots, source, **kwargs):
        snapshots = list(snapshots)
        captured["snapshot"] = snapshots[0]
        captured["source"] = source
        return {snapshots[0].t: None}

    monkeypatch.setattr(
        "qraft.construction.policies.allocation.build_optimizer_input_table",
        fake_build_optimizer_input_table,
    )

    run = Allocation(
        market,
        EqualWeightPolicy(target_cash_weight=0.2),
        forecasts=forecasts,
    ).at()

    assert run.forecasts is forecasts
    assert run.projection is not None
    assert run.optimizer_inputs is None
    assert run.as_of == datetime(2024, 1, 2)
    assert captured["source"].forecast_at(datetime(2024, 1, 2)) is forecasts
    assert captured["snapshot"].t == datetime(2024, 1, 2)
    assert captured["snapshot"].t_next == datetime(2024, 1, 2)


def test_policy_run_risk_reuses_existing_forecasts(monkeypatch) -> None:
    forecasts = _forecasts()
    run = run_policy(
        EqualWeightPolicy(target_cash_weight=0.2),
        _state(),
        forecasts,
    )
    captured = {}

    def fake_risk(self, forecasts_arg, **kwargs):
        captured["forecasts"] = forecasts_arg
        captured["kwargs"] = kwargs
        return "risk"

    monkeypatch.setattr(
        "qraft.construction.policies.policy_projection.PolicyProjection.risk",
        fake_risk,
    )

    risk = run.risk(auto_select_factors=True, criterion="bic")

    assert risk == "risk"
    assert captured["forecasts"] is forecasts
    assert captured["kwargs"] == {"auto_select_factors": True, "criterion": "bic"}


def test_policy_run_risk_supports_factors_free_universe() -> None:
    run = run_policy(
        EqualWeightPolicy(target_cash_weight=0.2),
        _state(),
        _forecasts(),
    )

    risk = run.risk()

    assert risk.r2 >= 0.0
    assert risk.risk_attribution.exposures == {"idiosyncratic": -1.0}
    assert risk.effective_bets().effective_bets == 1.0
