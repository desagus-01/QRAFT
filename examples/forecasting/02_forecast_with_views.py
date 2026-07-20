# %%
import logging

import numpy as np

from qraft import (
    Forecaster,
    PipelineConfig,
    SimulationForecastConfig,
    Views,
)
from qraft.core.scenarios.view_types import MeanView, RankingView
from qraft.core.scenarios.views import ViewWindow
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_market_example

setup_logging(LogConfig(level=logging.INFO))

# %%
# Build a baseline market with STRESS_INDEX, GROWTH_PULSE, and RATE_WAVE as non-tradable factors.
market_without_views = synthetic_market_example()

# Use the latest quarter-end forecast date that has a following bar available.
as_of = market_without_views.trading_bars[-89]

# %%
# Express a risk-off view at the forecast date. The STRESS_INDEX constraint says the
# next scenario should look like a high-stress historical move, while the ranking
# view says defensive assets should outperform equities.
returns = market_without_views.returns_through(as_of)
high_stress_move = float(
    np.quantile(returns.values.get_column("STRESS_INDEX").to_numpy(), 0.80)
)

risk_off_view = ViewWindow(
    start=as_of,
    end=as_of,
    views=Views(
        [
            MeanView("STRESS_INDEX", ">=", high_stress_move),
            RankingView(["ORION", "TERRA", "CYPHER", "LUMEN", "NOVA"]),
        ],
        confidence=0.75,
    ),
    name="risk_off_latest_forecast",
)
market_with_views = market_without_views.with_views([risk_off_view])

# %%
# Inspect how the view reweights historical scenarios before forecasting.
view_report = market_with_views.view_report(as_of)
print(view_report.diagnostics)

view_report.plot(prob_mode="regular")
# %%
# Keep the same recipe and simulation settings so differences come from the
# active view's posterior scenario probabilities rather than model selection.
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=21, n_sims=500, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

recipes = forecaster.recipes(market_without_views, min_history=180)
baseline_run = forecaster.run(
    market_without_views,
    min_history=180,
    forecast_cadence="quarter_end",
    recipes=recipes,
)
viewed_run = forecaster.run(
    market_with_views,
    min_history=180,
    forecast_cadence="quarter_end",
    recipes=recipes,
)

# %%
# Compare the same forecast date in both runs. The viewed run should use the
# posterior scenario probabilities at as_of because the view window is active.
viewed_forecast = viewed_run.forecast_at(as_of)
print(viewed_forecast.at_step(1, subset="tradable").values.head())

# %%
# Plot the viewed forecast paths.
viewed_forecast.plot_asset_paths(subset="tradable", max_assets=5, ncols=3)
