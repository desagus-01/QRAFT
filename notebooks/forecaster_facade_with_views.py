# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    Forecaster,
    Prior,
    MarketData,
    PipelineConfig,
    SimulationForecastConfig,
    Views,
)
from qraft.core import CMAConfig
from qraft.core.scenarios.view_types import MeanView, QuantileView, RankingView, StdView
from qraft.core.scenarios.views import ViewWindow
from qraft.utils import LogConfig, setup_logging
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.INFO))

# %%
# Load sample data and build a causal market.
prices, factor_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
prices = prices.with_columns(pl.selectors.numeric().exp())
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

min_price = 12
cols_to_keep = [
    col
    for col in prices.columns
    if col == "date"
    or (
        col != "DHIL"
        and prices[col].null_count() == 0
        and prices[col].dtype.is_numeric()
        and float(prices[col].min()) >= min_price  # type: ignore[arg-type]
    )
]
prices = prices.select(cols_to_keep)

assets = list(prices.columns[10:20])
universe = AssetUniverse(assets=assets, factors=list(factor_cols)[:4])
prices = prices.select("date", *universe.all_tickers)

market_without_views = MarketData.from_prices(
    prices,
    universe,
    cash=cash,
    prior=Prior.time_conditioned(half_life=60),
)

# %%
# Register time-local views with explicit active windows.
view_asset = assets[0]
view_dates = [prices["date"][-360], prices["date"][-240], prices["date"][-120]]
date_list = prices["date"].to_list()
view_ends = [
    date_list[date_list.index(view_dates[1]) - 1],
    date_list[date_list.index(view_dates[2]) - 1],
    prices["date"][-1],
]
return_panels = [
    market_without_views._simple_return_history_through(view_date)
    for view_date in view_dates
]
normal_averages = [
    np.average(panel.values.get_column(view_asset).to_numpy(), weights=panel.prob)
    for panel in return_panels
]

view_windows = [
    ViewWindow(
        view_dates[0],
        view_ends[0],
        Views(
            [
                MeanView(view_asset, ">=", normal_averages[0] * 1.25),
                RankingView(order=assets[:3]),
            ],
            confidence=0.55,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
    ViewWindow(
        view_dates[1],
        view_ends[1],
        Views(
            [
                MeanView(assets[1], "<=", normal_averages[1] * 0.75),
                RankingView(order=[assets[2], assets[0], assets[1]]),
            ],
            confidence=0.75,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
    ViewWindow(
        view_dates[2],
        view_ends[2],
        Views(
            [
                MeanView(view_asset, "==", normal_averages[2] * 1.10),
                StdView(view_asset, "<=", 0.035),
                QuantileView(view_asset, 0.25, 0.20),
            ],
            confidence=0.90,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
]
market = market_without_views.with_views(view_windows)

# %%
# Forecaster facade: create recipes, then create a forecast run from those recipes.
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(
        horizon=10,
        method="cma",
        n_sims=10_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
    new_recipe_every=int(prices.height / 4),
    new_recipe_cadence="every_bar",
    seed=10,
)

min_history = 252
recipes = forecaster.recipes(market, min_history=min_history)
run = forecaster.run(
    market,
    min_history=min_history,
    forecast_cadence="quarter_end",
    recipes=recipes,
)

# %%
# High-level run diagnostics.
recipe_summary = pl.DataFrame(
    {
        "period": list(range(len(recipes.periods))),
        "start": [period.start for period in recipes.periods],
        "end": [period.end for period in recipes.periods],
        "selected_at_step": [period.selected_at_step for period in recipes.periods],
        "n_invariance_drops": [
            len(period.invariance_drops) for period in recipes.periods
        ],
    }
)
step_summary = pl.DataFrame(
    {
        "as_of": [step.as_of for step in run.steps],
        "recipe_period": [step.recipe_period_index for step in run.steps],
        "action": [step.action for step in run.steps],
        "n_invariance_drops": [len(step.invariance_drops) for step in run.steps],
    }
)
model_health = run.model_health_df()

recipe_summary

# %%
step_summary

# %%
model_health

# %%
# Inspect diagnostics from the latest forecast step.
latest_step = run.steps[-1]
latest_forecast = latest_step.forecast
latest_step.as_of, latest_step.action, latest_step.recipe_period_index

# %%
latest_step.diagnostics

# %%
latest_step.invariance_drops

# %%
latest_forecast.model_health_df()

# %%
# Plot simulated paths for the latest forecast.
fig_paths = latest_forecast.plot_asset_paths(
    subset="tradable",
    assets=assets[:6],
    max_assets=6,
    ncols=3,
)
fig_paths

# %%
# Plot paths for a forecast immediately after the last view date, if available.
viewed_steps = [step for step in run.steps if step.as_of >= view_dates[-1]]
viewed_step = viewed_steps[0] if viewed_steps else latest_step
fig_viewed_paths = viewed_step.forecast.plot_asset_paths(
    subset="tradable",
    assets=assets[:6],
    max_assets=6,
    ncols=3,
)
fig_viewed_paths
