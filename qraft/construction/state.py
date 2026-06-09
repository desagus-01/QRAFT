from dataclasses import dataclass

import numpy as np
from qraft.forecast.forecast_paths import ForecastPaths
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PortfolioState:
    asset_order: list[str]
    initial_prices: NDArray[np.floating]
    shares: NDArray[np.int32]
    cash: float

    def __post_init__(self) -> None:
        if self.shares.shape[0] != len(self.asset_order):
            raise ValueError(
                "Number of shares do NOT match with the number of assets you have"
            )
        if self.initial_prices.shape[0] != len(self.asset_order):
            raise ValueError(
                "Number of initial prices do NOT match with the number of assets you have"
            )

    @classmethod
    def from_forecast_and_assets(
        cls,
        asset_forecasts: ForecastPaths,
        assets: list[str],
        shares: NDArray[np.int32],
        cash: float,
    ):
        if asset_forecasts.universe is None:
            raise ValueError(
                "ForecastPaths.universe must be set to construct a PortfolioState"
            )

        initial_prices = np.asarray(
            [asset_forecasts.initial_prices[asset] for asset in assets]
        )

        return cls(
            asset_order=assets,
            initial_prices=initial_prices,
            shares=shares,
            cash=cash,
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
    def cash_weight(self) -> NDArray[np.floating]:
        return self.cash / self.portfolio_value

    @property
    def asset_weights(self) -> NDArray[np.floating]:
        values = self.initial_prices * self.shares
        return values / self.portfolio_value

    @property
    def portfolio_weights_dict(self) -> dict[str, NDArray[np.floating]]:
        values = self.initial_prices * self.shares
        values_inc_cash = np.append(values, self.cash)
        assets_inc_cash = self.asset_order + ["cash"]
        weights = values_inc_cash / self.portfolio_value
        return dict(zip(assets_inc_cash, weights))
