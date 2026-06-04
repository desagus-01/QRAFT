from dataclasses import dataclass
from typing import Literal

import numpy as np
from construction.policies import PolicyDecision
from construction.state import PortfolioState
from forecast.forecast_paths import ForecastPaths
from forecast.scenarios.types import ProbVector
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


@dataclass(frozen=True, slots=True)
class PolicyProjection:
    initial_value: NDArray[np.floating]
    forecast_values: NDArray[np.floating]
    path_probs: ProbVector

    @classmethod
    def from_decision(
        cls,
        decision: PolicyDecision,
        forecasts: ForecastPaths,
        state: PortfolioState,
    ):
        if decision.cash_return is None:
            raise ValueError(
                "PolicyDecision.cash_return is required to project policy forecasts"
            )

        assets = decision.asset_order
        initial_prices = np.array([forecasts.initial_prices[asset] for asset in assets])
        allocated_shares = (
            decision.target_weights_risk * state.portfolio_value
        ) / initial_prices
        price_stack = forecasts.price_stack_for(assets=assets)
        portfolio_forecasts = np.einsum(
            "a,aph->ph",
            allocated_shares,
            price_stack,
        )

        cash_forecasts = create_cash_forecasts(
            cash_return=decision.cash_return,
            cash_weight=decision.target_cash_weight,
            portfolio_value=state.portfolio_value,
            forecast_shape=(forecasts.n_simulations, forecasts.n_horizons),
        )

        return cls(
            initial_value=state.portfolio_value,
            forecast_values=portfolio_forecasts + cash_forecasts,
            path_probs=forecasts.path_probs,
        )

    @property
    def cumulative_returns(self) -> NDArray[np.floating]:
        return self.forecast_values / self.initial_value - 1

    def performance_at_period(self, period: int) -> NDArray[np.floating]:
        return self.cumulative_returns[:, period]

    def plot(self, type: Literal["value", "cum_performance"]) -> None:
        value_to_plot = (
            self.forecast_values if type == "value" else self.cumulative_returns
        )

        return plot_simulation_results(value_to_plot, title=f"Portfolio {type}")
