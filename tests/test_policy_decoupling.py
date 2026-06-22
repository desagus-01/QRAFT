from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.construction.market_snapshot import MarketSnapshot
from qraft.construction.optimization.moments import PolicyInputConfig, PolicyInputs
from qraft.construction.optimization.objectives.specs import ExpectedReturn
from qraft.construction.optimization.problem import MPOProblemBuilder
from qraft.construction.policies import ForecastingMPOPolicy, MPOPolicy
from qraft.construction.state import PortfolioState
from qraft.core.panel import ScenarioPanel
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths


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


def test_forecasting_mpo_policy_is_explicit_orchestrator(monkeypatch, tmp_path) -> None:
    problem = MPOProblemBuilder().add(ExpectedReturn()).build()
    cash_path = tmp_path / "cash.csv"
    cash_path.write_text("date,DFF\n2024-01-02,0.0\n")
    policy = ForecastingMPOPolicy(
        policy=MPOPolicy(problem=problem),
        input_config=PolicyInputConfig(cash_path=str(cash_path)),
        min_history=0,
    )

    forecasts = ForecastPaths(
        asset_paths={"A": np.array([[10.1], [10.2]])},
        dates=pl.Series("date", [datetime(2024, 1, 3)]),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 10.0},
        universe=AssetUniverse.factors_free(["A"]),
    )
    monkeypatch.setattr(
        "qraft.construction.policies.forecasting.run_forecast",
        lambda **kwargs: forecasts,
    )

    decision = policy.decide(_snapshot(), _state())

    assert decision.asset_order == ["A"]
    assert decision.diagnostics.is_optimal
