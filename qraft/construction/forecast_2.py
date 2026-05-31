from dataclasses import dataclass

import numpy as np
from construction.policies import PolicyDecision
from forecast.pipelines.forecasting import ForecastPaths
from numpy.typing import NDArray


# This needs to be created from policy + MC forecast...
@dataclass(frozen=True, slots=True)
class PortfolioExecution:
    forecast_values: NDArray[np.floating]
    rebalance_frequency: int

    @classmethod
    def from_policy_and_forecasts(
        cls,
        policy_decision: PolicyDecision,
        forecasts: ForecastPaths,
        rebalance_frequency: int,
    ):
        portfolio_forecasts = np.einsum(
            "a,aph->ph", policy_decision.target_weights_risk, forecasts.price_stack
        )

        return cls(
            forecast_values=portfolio_forecasts,
            rebalance_frequency=rebalance_frequency,
        )
