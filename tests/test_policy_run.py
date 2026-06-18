import numpy as np

from qraft.construction.policies import EqualWeightPolicy
from qraft.construction.policy_decision import PolicyDecision
from qraft.construction.policy_projection import PolicyProjection
from qraft.construction.policy_run import PolicyRun, run_policy
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import ForecastPaths


def _forecasts() -> ForecastPaths:
    return ForecastPaths(
        asset_paths={
            "A": np.array([[11.0, 12.0], [9.0, 10.0]]),
            "B": np.array([[22.0, 24.0], [18.0, 20.0]]),
        },
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 10.0, "B": 20.0},
    )


def _state() -> PortfolioState:
    return PortfolioState(
        asset_order=["A", "B"],
        initial_prices=np.array([10.0, 20.0]),
        shares=np.array([0, 0], dtype=np.int32),
        cash=100.0,
    )


def test_policy_decide_returns_decision_without_projecting() -> None:
    decision = EqualWeightPolicy(target_cash_weight=0.2).decide(_state(), _forecasts())

    assert isinstance(decision, PolicyDecision)
    assert decision.asset_order == ["A", "B"]
    np.testing.assert_allclose(decision.target_weights_risk, [0.4, 0.4])
    assert decision.target_cash_weight == 0.2


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
    result = run_policy(
        EqualWeightPolicy(target_cash_weight=0.2),
        _state(),
        _forecasts(),
    )

    assert isinstance(result, PolicyRun)
    assert isinstance(result.decision, PolicyDecision)
    assert isinstance(result.projection, PolicyProjection)
    assert result.projection.target_cash_weight == result.decision.target_cash_weight
    np.testing.assert_allclose(
        result.projection.target_weights_risk,
        result.decision.target_weights_risk,
    )
