import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qraft.forecast.forecast_paths import ForecastPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PortfolioState:
    asset_order: list[str]
    initial_prices: NDArray[np.floating]
    shares: NDArray[np.floating]
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
        if not np.all(np.isfinite(self.initial_prices)):
            raise ValueError("initial_prices must contain only finite values")
        if np.any(self.initial_prices <= 0):
            raise ValueError("initial_prices must be strictly positive")
        if not np.all(np.isfinite(self.shares)):
            raise ValueError("shares must contain only finite values")
        if not np.isfinite(self.cash):
            raise ValueError("cash must be finite")
        if not np.isfinite(self.portfolio_value) or self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be finite and strictly positive")

    @classmethod
    def from_forecast_and_assets(
        cls,
        asset_forecasts: ForecastPaths,
        assets: list[str],
        shares: NDArray[np.floating],
        cash: float,
    ):
        if asset_forecasts.universe is None:
            raise ValueError(
                "ForecastPaths.universe must be set to construct a PortfolioState"
            )

        forecast_assets = list(asset_forecasts.initial_prices.keys())
        kept, missing = cls._resolve_assets(assets, forecast_assets)

        if missing:
            logger.warning(
                "Building PortfolioState: %d asset(s) not in forecast — "
                "omitted from state: %s",
                len(missing),
                missing,
            )
            idx_map = [assets.index(a) for a in kept]
            shares = shares[idx_map]

        initial_prices = np.asarray(
            [asset_forecasts.initial_prices[asset] for asset in kept]
        )

        return cls(
            asset_order=kept,
            initial_prices=initial_prices,
            shares=shares,
            cash=cash,
        )

    @classmethod
    def from_cash(
        cls,
        cash: float,
        assets: list[str],
        asset_forecasts: ForecastPaths,
    ):
        if asset_forecasts.universe is None:
            raise ValueError(
                "ForecastPaths.universe must be set to construct a PortfolioState"
            )

        forecast_assets = list(asset_forecasts.initial_prices.keys())
        kept, missing = cls._resolve_assets(assets, forecast_assets)

        if missing:
            logger.warning(
                "Building PortfolioState: %d asset(s) not in forecast — "
                "omitted from state: %s",
                len(missing),
                missing,
            )

        initial_prices = np.asarray(
            [asset_forecasts.initial_prices[asset] for asset in kept]
        )

        return cls(
            asset_order=kept,
            initial_prices=initial_prices,
            shares=np.zeros(len(kept), dtype=float),
            cash=cash,
        )

    @staticmethod
    def _resolve_assets(
        requested: list[str],
        available: list[str],
    ) -> tuple[list[str], list[str]]:
        avail_set = set(available)
        kept = [a for a in requested if a in avail_set]
        missing = [a for a in requested if a not in avail_set]
        return kept, missing

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


def align_state_to_assets(
    state: PortfolioState,
    kept_assets: list[str],
) -> tuple[NDArray[np.floating], float, list[str], float]:
    weights_by_asset = state.portfolio_weights_dict
    kept_set = set(kept_assets)
    dropped = [a for a in state.asset_order if a not in kept_set]
    dropped_weight = sum(float(weights_by_asset[a]) for a in dropped)
    current_cash = float(state.cash_weight) + dropped_weight
    current_weights = np.array(
        [float(weights_by_asset[a]) for a in kept_assets], dtype=float
    )
    return current_weights, current_cash, dropped, dropped_weight
