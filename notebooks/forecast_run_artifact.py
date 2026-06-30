# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    CMAConfig,
    LogConfig,
    PipelineConfig,
    SimulationForecastConfig,
    setup_logging,
)
from qraft.backtest.market import MarketData, WindowWeighting
from qraft.forecast.run import build_forecast_recipe_history, simulate_forecast_paths
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# %%
# Load the same sample data used by the other notebook scripts.
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

min_price = 15
cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= np.log(min_price)  # type: ignore[arg-type]
    )
]
data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:20])

# %%
universe = AssetUniverse(assets=tradable_assets, factors=list(factors_cols))
data = data.select("date", *universe.all_tickers)

market = MarketData.from_log_prices(
    data,
    universe,
    cash=cash,
    weighting=WindowWeighting("state_smooth", half_life=60),
)


# %%
recipe_history = build_forecast_recipe_history(
    market,
    min_history=150,
    refit_every=22,
    seed=3,
    pipeline_config=PipelineConfig(exclude_non_invariants=False),
)

run = simulate_forecast_paths(
    market,
    recipe_history,
    min_history=250,
    forecast_cadence="quarter_end",
    simulation_config=SimulationForecastConfig(
        horizon=15,
        method="cma",
        n_sims=10_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
)
# %%
len(run.recipe_history.periods)
