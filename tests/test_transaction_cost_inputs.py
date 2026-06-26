import numpy as np
import pytest

from qraft.backtest.inputs import ForecastInputsProvider
from qraft.construction.optimization.moments import PolicyInputConfig
from qraft.construction.optimization.moments import PolicyInputs
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


def _cvar_inputs_without_covariance() -> PolicyInputs:
    return PolicyInputs.from_arrays(
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


def test_forecast_provider_infers_both_for_cvar_with_market_impact() -> None:
    policy = MPOPolicy.preset(
        objective_type="cvar_cuts",
        risk_aversion=0.01,
        transaction_cost=TransactionCost(cost=0.0005, market_impact=0.3),
    )
    provider = ForecastInputsProvider(
        PolicyInputConfig(cash_path="data/cash.csv"),
        policy=policy,
    )

    assert provider._risk_source() == "both"


def test_forecast_provider_infers_cvar_for_cvar_without_market_impact() -> None:
    policy = MPOPolicy.preset(
        objective_type="cvar_cuts",
        risk_aversion=0.01,
        transaction_cost=TransactionCost(cost=0.0005, market_impact=0.0),
    )
    provider = ForecastInputsProvider(
        PolicyInputConfig(cash_path="data/cash.csv"),
        policy=policy,
    )

    assert provider._risk_source() == "cvar"


def test_forecast_provider_rejects_explicit_risk_missing_policy_requirements() -> None:
    policy = MPOPolicy.preset(
        objective_type="cvar_cuts",
        risk_aversion=0.01,
        transaction_cost=TransactionCost(cost=0.0005, market_impact=0.3),
    )
    provider = ForecastInputsProvider(
        PolicyInputConfig(cash_path="data/cash.csv", risk="cvar"),
        policy=policy,
    )

    with pytest.raises(ValueError, match="missing covariances"):
        provider._risk_source()
