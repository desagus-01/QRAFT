from typing import Sequence

from polars.dataframe.frame import DataFrame

from time_series.tests.seasonality import (
    SEASONAL_PERIODS,
    SeasonalityPeriodTest,
    periodogram_seasonality_test,
)


def seasonality_diagnostic(
    data: DataFrame,
    assets: list[str],
    *,
    seasonal_periods: Sequence[SEASONAL_PERIODS] | None = None,
) -> dict[str, list[SeasonalityPeriodTest]]:
    if seasonal_periods is None:
        seasonal_periods = ["weekly", "monthly", "quarterly", "semi-annual", "annual"]

    asset_seasonality_res = {}
    for asset in assets:
        data_array = (
            data.select(asset).drop_nulls().to_numpy().flatten()
        )  # we drop any nulls if there are any for that asset
        seasonal_res = [
            periodogram_seasonality_test(
                data=data_array, seasonal_period=seasonal_period
            )
            for seasonal_period in seasonal_periods
        ]
        asset_seasonality_res[asset] = seasonal_res

    return asset_seasonality_res
