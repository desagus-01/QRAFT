# %%
import logging

from qraft import (
    Forecaster,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_market_example

setup_logging(LogConfig(level=logging.INFO))
# %%
# Build a small synthetic market with five tradable assets and STRESS_INDEX, GROWTH_PULSE, and RATE_WAVE as factors.
market = synthetic_market_example()

# %%
# Configure the high-level forecasting facade.
#
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, n_sims=250, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

# %%
# Run forecasts after enough history is available. The forecast cadence controls
# which dates receive forecasts; new_recipe_cadence and new_recipe_every control
# when recipes are selected.
run = forecaster.run(
    market,
    min_history=180,
    forecast_cadence="quarter_end",
)

# %%
# Inspect selected models and diagnostics for each forecast step.
model_health = run.model_health_df()
print(model_health)

# %%
# Pull the latest forecast.
latest_step = run.steps[-1]
latest_forecast = latest_step.forecast

# %%
# ForecastPaths can also be viewed as a scenario panel for one forecast step.
one_step_panel = latest_forecast.at_step(1, subset="tradable")
print(one_step_panel.values.head())

# %%
# Plot simulated price paths for the tradable assets.
latest_forecast.plot_asset_paths(subset="tradable", max_assets=5, ncols=3)
