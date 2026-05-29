from dataclasses import dataclass

import numpy as np
from forecast.pipelines.forecasting import ForecastPaths
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PortfolioState:
    asset_order: list[str]
    initial_prices: NDArray[np.floating]
    shares: NDArray[np.int32]
    cash: float

    @classmethod
    def from_forecast(
        cls, asset_forecasts: ForecastPaths, shares: NDArray[np.int32], cash: float
    ):
        if asset_forecasts.universe is None:
            raise ValueError(
                "ForecastPaths.universe must be set to construct a PortfolioState"
            )

        tradable_assets = asset_forecasts.universe.assets
        initial_prices = np.asarray(
            [asset_forecasts.initial_prices[asset] for asset in tradable_assets]
        )

        return cls(
            asset_order=tradable_assets,
            initial_prices=initial_prices,
            shares=shares,
            cash=cash,
        )

    def __post_init__(self) -> None:
        if self.shares.shape[0] != len(self.asset_order):
            raise ValueError(
                "Number of shares do NOT match with the number of assets you have"
            )
        if self.initial_prices.shape[0] != len(self.asset_order):
            raise ValueError(
                "Number of initial prices do NOT match with the number of assets you have"
            )

    @property
    def initial_prices_dict(self) -> dict[str, NDArray[np.floating]]:
        return dict(zip(self.asset_order, self.initial_prices))

    @property
    def shares_dict(self) -> dict[str, NDArray[np.int32]]:
        return dict(zip(self.asset_order, self.shares))

    @property
    def asset_values(self) -> dict[str, NDArray[np.floating]]:
        values = self.initial_prices * self.shares
        return dict(zip(self.asset_order, values))

    @property
    def portfolio_value(self) -> NDArray[np.floating]:
        return np.dot(self.initial_prices, self.shares) + self.cash

    @property
    def portfolio_weights(self) -> dict[str, NDArray[np.floating]]:
        values = self.initial_prices * self.shares
        values_inc_cash = np.append(values, self.cash)
        assets_inc_cash = self.asset_order + ["cash"]
        weights = values_inc_cash / self.portfolio_value
        return dict(zip(assets_inc_cash, weights))
