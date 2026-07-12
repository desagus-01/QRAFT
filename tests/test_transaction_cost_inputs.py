import numpy as np
import pytest

from qraft.construction.inputs import required_optimizer_inputs
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.optimization.objectives.specs import TransactionCost
from qraft.construction.policies import MPOPolicy
from qraft.construction.state import PortfolioState


def _state() -> PortfolioState:
    return PortfolioState(
        asset_order=["A"],
        initial_prices=np.array([10.0]),
        shares=np.array([0.0]),
        cash=100.0,
    )


def _cvar_inputs_without_covariance() -> OptimizerInputs:
    return OptimizerInputs.from_arrays(
        assets=["A"],
        mean=np.array([[0.01]]),
        scenario_returns=np.array([[[0.02]], [[-0.01]], [[0.005]]]),
        scenario_probs=np.array([1 / 3, 1 / 3, 1 / 3]),
        cash_return=np.array([0.0]),
    )


def test_cvar_cuts_with_linear_transaction_cost_does_not_need_covariance() -> None:
    policy = MPOPolicy.preset(
        objective_type="cvar_cuts",
        risk_aversion=0.01,
        transaction_cost=TransactionCost(cost=0.0005, market_impact=0.0),
    )

    decision = policy.optimize(_state(), _cvar_inputs_without_covariance())

    assert decision.diagnostics.is_optimal


def test_transaction_cost_market_impact_requires_covariance() -> None:
    policy = MPOPolicy.preset(
        objective_type="cvar_cuts",
        risk_aversion=0.01,
        transaction_cost=TransactionCost(cost=0.0005, market_impact=0.3),
    )

    with pytest.raises(ValueError, match="TransactionCost.market_impact requires"):
        policy.optimize(_state(), _cvar_inputs_without_covariance())


def test_policy_without_risk_aversion_fails_when_run_directly() -> None:
    policy = MPOPolicy.preset("cvar_cuts")

    with pytest.raises(ValueError, match="risk_aversion"):
        policy.optimize(_state(), _cvar_inputs_without_covariance())


def test_forecast_provider_infers_both_for_cvar_with_market_impact() -> None:
    policy = MPOPolicy.preset(
        objective_type="cvar_cuts",
        risk_aversion=0.01,
        transaction_cost=TransactionCost(cost=0.0005, market_impact=0.3),
    )
    assert required_optimizer_inputs(policy).risk_source == "both"


def test_forecast_provider_infers_cvar_for_cvar_without_market_impact() -> None:
    policy = MPOPolicy.preset(
        objective_type="cvar_cuts",
        risk_aversion=0.01,
        transaction_cost=TransactionCost(cost=0.0005, market_impact=0.0),
    )
    assert required_optimizer_inputs(policy).risk_source == "cvar"


def test_forecast_provider_has_no_explicit_risk_override() -> None:
    policy = MPOPolicy.preset(
        objective_type="cvar_cuts",
        risk_aversion=0.01,
        transaction_cost=TransactionCost(cost=0.0005, market_impact=0.3),
    )
    required = required_optimizer_inputs(policy)

    assert required.covariances
    assert required.scenarios
