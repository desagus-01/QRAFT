# %%
import logging
from collections import defaultdict

import polars as pl

from qraft import (
    AssetUniverse,
    LogConfig,
    PipelineConfig,
    setup_logging,
)
from qraft.core.market import MarketData, Prior
from qraft.forecast.run import build_forecast_recipe_history
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# %%
# Load the same sample data used by the other notebook scripts.
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
data = data.with_columns(pl.selectors.numeric().exp())
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

min_price = 15
cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= min_price  # type: ignore[arg-type]
    )
]
data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:20])

# %%
universe = AssetUniverse(assets=tradable_assets, factors=list(factors_cols))
data = data.select("date", *universe.all_tickers)

market = MarketData.from_prices(
    data,
    universe,
    cash=cash,
    prior=Prior.time_conditioned(half_life=60),
)


# %%
recipe_history = build_forecast_recipe_history(
    market,
    min_history=150,
    new_recipe_every=int(data.height / 5),
    new_recipe_cadence="every_bar",
    seed=3,
    pipeline_config=PipelineConfig(exclude_non_invariants=False),
)
# %%

x = recipe_history.periods
repeat_stats = defaultdict(
    lambda: {
        "same_mean_count": 0,
        # "same_mean_dates": [],
        "same_vol_count": 0,
        # "same_vol_dates": [],
        "same_mean_and_vol_count": 0,
        # "same_mean_and_vol_dates": [],
        # "same_mean_consecutive_dates": [],
        # "same_vol_consecutive_dates": [],
        # "same_mean_and_vol_consecutive_dates": [],
    }
)

previous_by_asset = {}

for period_index, period in enumerate(x):
    date = period.end
    mean_orders = period.recipe.mean_orders
    vol_orders = period.recipe.vol_orders

    assets = set(mean_orders) | set(vol_orders)

    for asset in assets:
        mean_order = mean_orders.get(asset)
        vol_order = vol_orders.get(asset)

        previous = previous_by_asset.get(asset)

        if previous is not None:
            same_mean = mean_order == previous["mean_order"]
            same_vol = vol_order == previous["vol_order"]

            consecutive = period_index == previous["period_index"] + 1

            if same_mean:
                repeat_stats[asset]["same_mean_count"] += 1
                # repeat_stats[asset]["same_mean_dates"].append(date)

                # if consecutive:
                #     repeat_stats[asset]["same_mean_consecutive_dates"].append(date)
                #
            if same_vol:
                repeat_stats[asset]["same_vol_count"] += 1
                # repeat_stats[asset]["same_vol_dates"].append(date)

                # if consecutive:
                #     repeat_stats[asset]["same_vol_consecutive_dates"].append(date)
                #
            if same_mean and same_vol:
                repeat_stats[asset]["same_mean_and_vol_count"] += 1
                # repeat_stats[asset]["same_mean_and_vol_dates"].append(date)

                # if consecutive:
                #     repeat_stats[asset]["same_mean_and_vol_consecutive_dates"].append(
                #         date
                #     )
                #
        previous_by_asset[asset] = {
            "period_index": period_index,
            "date": date,
            "mean_order": mean_order,
            "vol_order": vol_order,
        }

repeat_stats = dict(repeat_stats)
repeat_stats
