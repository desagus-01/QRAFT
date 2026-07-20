# %%
"""Construction with a preset multi-period optimization policy."""

import logging

from qraft import (
    Allocation,
    Forecaster,
    MPOPolicy,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.construction import LongOnly, MaxWeight, MinCashWeight, TurnoverLimit
from qraft.construction.optimization import InputPlan
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_vix_market

setup_logging(LogConfig(level=logging.INFO))

# %%
# The construction module consumes a market, forecasts, a portfolio state, and a
# policy. Allocation builds the default cash-only state and optimizer inputs.
market = synthetic_vix_market()

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, n_sims=500, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

# %%
# InputPlan says how forecast paths should be converted into optimizer inputs.
# This policy uses forecast expected returns across up to 10 horizons.
plan = InputPlan(expected_returns="forecast", max_horizons=10)

policy = MPOPolicy.preset(
    objective_type="mean_covariance",
    risk_aversion=5.0,
    constraints=(
        LongOnly(),
        MinCashWeight(0.05),
        MaxWeight(0.60),
        TurnoverLimit(0.25),
    ),
    input_plan=plan,
    min_history=180,
    name="mean_covariance_example",
)

# %%
# Only the first optimized horizon is actionable. Later horizons are retained in
# diagnostics so you can inspect the optimizer's plan.
run = Allocation(
    market=market,
    policy=policy,
    forecasts=forecaster,
    initial_cash=100_000,
).at()

print(run.target_weights)
print({"cash": run.decision.target_cash_weight})

# %%
# Optimizer diagnostics are baked into the run. Mean-covariance policies require
# covariance inputs, so plan metrics are available without scenario CVaR inputs.
print(run.plan_metrics())

result = run.mpo_result()
print({"solver_status": result.status, "turnover": result.turnover})

# %%
run.plot_weights()
run.projection.plot()
