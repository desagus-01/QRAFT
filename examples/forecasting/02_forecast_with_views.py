# %%
"""Forecast simulation with scenario views."""

import logging

import numpy as np
import polars as pl

from qraft import (
    Forecaster,
    LogConfig,
    PipelineConfig,
    SimulationForecastConfig,
    Views,
    setup_logging,
)
from qraft.core.scenarios.view_types import MeanView, RankingView
from qraft.core.scenarios.views import ViewWindow
from qraft.utils.example_data import synthetic_vix_market

setup_logging(LogConfig(level=logging.INFO))

# %%
# Build a baseline market with VIX as a non-tradable factor.
market_without_views = synthetic_vix_market()

# Use the latest quarter-end forecast date that has a following bar available.
as_of = market_without_views.trading_bars[-89]

# %%
# Express a risk-off view at the forecast date. The VIX constraint says the
# next scenario should look like a high-VIX historical move, while the ranking
# view says defensive assets should outperform equities.
returns = market_without_views.returns_through(as_of)
high_vix_move = float(np.quantile(returns.values.get_column("VIX").to_numpy(), 0.80))

risk_off_view = ViewWindow(
    start=as_of,
    end=as_of,
    views=Views(
        [
            MeanView("VIX", ">=", high_vix_move),
            RankingView(["TLT", "GLD", "SPY"]),
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
baseline_forecast = baseline_run.forecast_at(as_of)
viewed_forecast = viewed_run.forecast_at(as_of)

rows = []
for asset in baseline_forecast.tradable_paths:
    baseline_terminal = (
        baseline_forecast.asset_paths[asset][:, -1]
        / baseline_forecast.initial_prices[asset]
        - 1.0
    )
    viewed_terminal = (
        viewed_forecast.asset_paths[asset][:, -1]
        / viewed_forecast.initial_prices[asset]
    ) - 1.0

    baseline_var_5 = float(np.quantile(baseline_terminal, 0.05))
    viewed_var_5 = float(np.quantile(viewed_terminal, 0.05))

    rows.append(
        {
            "asset": asset,
            "baseline_mean": float(
                np.average(baseline_terminal, weights=baseline_forecast.path_probs)
            ),
            "viewed_mean": float(
                np.average(viewed_terminal, weights=viewed_forecast.path_probs)
            ),
            "mean_shift": float(
                np.average(viewed_terminal, weights=viewed_forecast.path_probs)
                - np.average(baseline_terminal, weights=baseline_forecast.path_probs)
            ),
            "baseline_var_5": baseline_var_5,
            "viewed_var_5": viewed_var_5,
            "var_5_shift": viewed_var_5 - baseline_var_5,
            "baseline_prob_below_t0": float(
                np.average(
                    baseline_terminal < 0.0, weights=baseline_forecast.path_probs
                )
            ),
            "viewed_prob_below_t0": float(
                np.average(viewed_terminal < 0.0, weights=viewed_forecast.path_probs)
            ),
        }
    )

comparison = pl.DataFrame(rows)
print(comparison)

# %%
# Plot the viewed forecast paths. Compare with baseline_forecast.plot_asset_paths
# to see how the risk-off view changes the distribution.
viewed_forecast.plot_asset_paths(subset="tradable", max_assets=3, ncols=3)
baseline_forecast.plot_asset_paths(subset="tradable", max_assets=3, ncols=3)
