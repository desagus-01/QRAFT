from dataclasses import dataclass
from typing import Literal

import numpy as np
from construction.optimization.moments import HorizonMoments
from construction.policies import PolicyDecision
from construction.state import PortfolioState
from forecast.pipelines.forecasting import ForecastPaths
from numpy.typing import NDArray
from utils.visuals import plot_simulation_results


def create_cash_forecasts(
    cash_return: NDArray[np.floating],
    cash_weight: float,
    portfolio_value: NDArray[np.floating] | float,
    forecast_shape: tuple[int, int],
) -> NDArray[np.floating]:
    broadcast_cash = np.broadcast_to(
        cash_return, (forecast_shape[0], forecast_shape[1])
    )
    cash_allocation = cash_weight * portfolio_value
    cash_cumulative_growth = np.cumprod(1 + broadcast_cash, axis=1)
    return cash_allocation * cash_cumulative_growth


# This needs to be created from policy + MC forecast...
@dataclass(frozen=True, slots=True)
class PortfolioExecution:
    initial_value: NDArray[np.floating]
    forecast_values: NDArray[np.floating]

    @classmethod
    def from_policy_and_forecasts(
        cls,
        policy_decision: PolicyDecision,
        forecasts: ForecastPaths,
        state: PortfolioState,
        assets: list[str],
    ):
        initial_prices = np.array([forecasts.initial_prices[asset] for asset in assets])
        allocated_shares = (
            policy_decision.target_weights_risk * state.portfolio_value
        ) / initial_prices
        price_stack = forecasts.price_stack_for(assets=assets)
        portfolio_forecasts = np.einsum(
            "a,aph->ph",
            allocated_shares,
            price_stack,
        )

        cash_return = HorizonMoments.get_cash_return("data/cash.csv", 1)
        cash_forecasts = create_cash_forecasts(
            cash_return=cash_return,
            cash_weight=policy_decision.target_cash_weight,
            portfolio_value=state.portfolio_value,
            forecast_shape=(forecasts.n_simulations, forecasts.n_horizons),
        )

        return cls(
            initial_value=state.portfolio_value,
            forecast_values=portfolio_forecasts + cash_forecasts,
        )

    @property
    def cumulative_returns(self) -> NDArray[np.floating]:
        return self.forecast_values / self.initial_value - 1

    def plot(self, type: Literal["value", "cum_performance"]) -> None:
        value_to_plot = (
            self.forecast_values if type == "value" else self.cumulative_returns
        )

        return plot_simulation_results(value_to_plot, title=f"Portfolio {type}")
