from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import ForecastPaths


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
    decision_error: str | None = None


BacktestWarning = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    policy_name: str
    asset_order: list[str]
    nav_dates: list[datetime]
    nav: NDArray[np.floating]
    periods: list[BacktestPeriod]
    warnings_log: list[BacktestWarning] = field(default_factory=list)


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
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    asset_value = shares * prices
    nav = float(asset_value.sum() + cash)
    target_value = _target_weights_full(decision, asset_order) * nav
    trade_value = target_value - asset_value
    executed = trade_value / prices
    return executed, shares + executed, cash - float(trade_value.sum())
