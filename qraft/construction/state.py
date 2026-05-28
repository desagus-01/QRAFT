from dataclasses import dataclass
from typing import Sequence

import numpy as np
from forecast.pipelines.forecasting import ForecastPaths
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PortfolioState:
    asset_order: Sequence[str]
    initial_prices: NDArray[np.floating]
    shares: NDArray[np.floating]
    cash: float

    @classmethod
    def from_forecast(
        cls, asset_forecasts: ForecastPaths, shares: NDArray[np.floating], cash: float
    ):
        return cls(
            asset_order=asset_forecasts.asset_names,
            initial_prices=np.asarray(list(asset_forecasts.initial_prices.values())),
            shares=shares,
            cash=cash,
        )
