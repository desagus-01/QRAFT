from dataclasses import dataclass

import numpy as np

from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import ForecastPaths


@dataclass(frozen=True, slots=True)
class AllCashPolicy:
    name: str = "all_cash"

    def decide(self, state: PortfolioState, forecasts: ForecastPaths) -> PolicyDecision:
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=np.zeros(len(state.asset_order)),
            target_cash_weight=1.0,
        )


@dataclass(frozen=True, slots=True)
class NoTradePolicy:
    name: str = "no_trade"

    def decide(self, state: PortfolioState, forecasts: ForecastPaths) -> PolicyDecision:
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=state.asset_weights,  # == current -> zero turnover
            target_cash_weight=float(state.cash_weight),
        )
