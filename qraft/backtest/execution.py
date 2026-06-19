from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from qraft.backtest.market import MarketSnapshot, RealisedStep
from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.core.configs import PipelineConfig, SimulationForecastConfig
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths
from qraft.forecast.pipelines.forecasting import run_forecast


class Forecaster(Protocol):
    min_history: int

    def forecast(
        self, snapshot: MarketSnapshot, universe: AssetUniverse
    ) -> ForecastPaths: ...


@dataclass(frozen=True, slots=True)
class PipelineForecaster:
    simulation_config: SimulationForecastConfig
    pipeline_config: PipelineConfig | None = None
    seed: int | None = None
    min_history: int = (
        504  # >= 2 annual cycles; below this the seasonality test is degenerate
    )
    ...

    def forecast(
        self, snapshot: MarketSnapshot, universe: AssetUniverse
    ) -> ForecastPaths:
        return run_forecast(
            panel=snapshot.history,
            universe=universe,
            seed=self.seed,
            simulation_config=self.simulation_config,
            pipeline_config=self.pipeline_config,
        )


@dataclass(frozen=True, slots=True)
class BacktestPeriod:
    decision_bar: datetime
    execution_bar: datetime
    state_before: PortfolioState
    decision: PolicyDecision
    executed_share_trades: NDArray[np.floating]
    state_after: PortfolioState
    solver_status: str | None
    forecasts: ForecastPaths | None = None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    policy_name: str
    asset_order: list[str]
    nav_dates: list[datetime]
    nav: NDArray[np.floating]
    periods: list[BacktestPeriod]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    executed_share_trades: NDArray[np.floating]
    shares_after: NDArray[np.floating]
    cash_after: float


def _target_weights_full(
    decision: PolicyDecision, asset_order: list[str]
) -> NDArray[np.floating]:
    idx = {a: i for i, a in enumerate(asset_order)}
    w = np.zeros(len(asset_order))
    for a, wi in zip(decision.asset_order, decision.target_weights_risk):
        w[idx[a]] = wi  # assets absent from the decision stay 0 -> sold to cash
    return w


def execute_frictionless(
    decision: PolicyDecision,
    shares: NDArray[np.floating],
    cash: float,
    prices: NDArray[np.floating],
    asset_order: list[str],
) -> ExecutionResult:
    asset_value = shares * prices
    nav = float(asset_value.sum() + cash)
    target_value = _target_weights_full(decision, asset_order) * nav
    trade_value = target_value - asset_value
    executed = trade_value / prices
    return ExecutionResult(
        executed_share_trades=executed,
        shares_after=shares + executed,
        cash_after=cash - float(trade_value.sum()),  # self-financing plug
    )


def advance_holdings(
    shares: NDArray[np.floating], cash: float, realised: RealisedStep
) -> tuple[NDArray[np.floating], float, float]:
    """Frictionless: compound cash over the bar, mark holdings. Returns (shares, cash, pre_trade_nav)."""
    cash = cash * (1.0 + realised.cash_return)
    nav = float(shares @ realised.prices_next + cash)
    return shares, cash, nav
