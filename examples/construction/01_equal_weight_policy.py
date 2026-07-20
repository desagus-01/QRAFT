# %%
"""Construction quickstart with an equal-weight policy."""

import logging

from qraft import (
    Allocation,
    EqualWeightPolicy,
    Forecaster,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_vix_market

setup_logging(LogConfig(level=logging.INFO))

# %%
# Build a small synthetic market with three tradable assets and VIX as a factor.
market = synthetic_vix_market()

# %%
# Forecasts are still useful for simple policies because they let us project the
# chosen allocation through simulated future paths.
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, n_sims=250, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

# %%
# EqualWeightPolicy does not optimize. It simply spreads the risky allocation
# equally across the assets and leaves the requested cash weight untouched.
policy = EqualWeightPolicy(target_cash_weight=0.05)

run = Allocation(
    market=market,
    policy=policy,
    forecasts=forecaster,
    initial_cash=100_000,
).at()

# %%
# The decision is the target allocation to rebalance to now.
print(run.target_weights)
print({"cash": run.decision.target_cash_weight})

# %%
# Diagnostics are lighter for this baseline policy, but the projection is still
# available for forecast-based performance and risk inspection.
print(run.projection.total_target_weights)
print(run.projection.cumulative_returns[:, -1].mean())

# %%
run.plot_weights()
run.projection.plot()
