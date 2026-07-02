from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.construction.market_snapshot import MarketSnapshot
from qraft.construction.optimization.inputs import PolicyInputs
from qraft.construction.optimization.objectives.specs import ExpectedReturn
from qraft.construction.optimization.problem import MPOProblemBuilder
from qraft.construction.policies import MPOPolicy
from qraft.construction.state import PortfolioState
from qraft.core.panel import ScenarioPanel
from qraft.forecast.forecast_paths import AssetUniverse


def _state() -> PortfolioState:
    return PortfolioState(
        asset_order=["A"],
        initial_prices=np.array([10.0]),
        shares=np.array([0.0]),
        cash=100.0,
    )


def _snapshot() -> MarketSnapshot:
    history = ScenarioPanel.from_prices(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "A": [9.0, 10.0],
            }
        )
    )
    return MarketSnapshot(
        t=datetime(2024, 1, 2),
        t_next=datetime(2024, 1, 3),
        universe=AssetUniverse.factors_free(["A"]),
        history=history,
        prices_t=np.array([10.0]),
        cash_rate=0.0,
    )


def test_mpo_policy_requires_explicit_policy_inputs() -> None:
    problem = MPOProblemBuilder().add(ExpectedReturn()).build()
    policy = MPOPolicy(problem=problem)

    with pytest.raises(ValueError, match="explicit PolicyInputs"):
        policy.decide(_snapshot(), _state())


def test_mpo_policy_optimizes_user_supplied_policy_inputs() -> None:
    problem = MPOProblemBuilder().add(ExpectedReturn()).build()
    policy = MPOPolicy(problem=problem)
    inputs = PolicyInputs.from_arrays(
        assets=["A"],
        mean=np.array([[0.01]]),
        cash_return=np.array([0.0]),
    )

    decision = policy.optimize(_state(), inputs)

    assert decision.asset_order == ["A"]
    assert decision.diagnostics.is_optimal
