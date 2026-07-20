# %%
import logging

from qraft import (
    Allocation,
    Forecaster,
    MPOPolicy,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.construction import (
    LongOnly,
    MaxWeight,
    MinCashWeight,
    MPOProblemBuilder,
    TurnoverLimit,
)
from qraft.construction.optimization import (
    CashReturn,
    CovarianceRisk,
    ExpectedReturn,
    InputPlan,
    TransactionCost,
)
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_market_example

setup_logging(LogConfig(level=logging.INFO))

# %%
# Start from the same synthetic market so this example focuses on customization.
market = synthetic_market_example()

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, n_sims=500, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

# %%
# MPOProblemBuilder lets you explicitly choose the objective terms and weights.
# Positive-return terms are rewarded; risk and cost terms are penalized.
problem = (
    MPOProblemBuilder()
    .add(ExpectedReturn(decay=0.9), weight=1.0)
    .add(CashReturn(), weight=1.0)
    .add(CovarianceRisk(), weight=4.0)
    .add(TransactionCost(cost=0.0005, pershare_cost=0.005), weight=1.0)
    .constrain(
        LongOnly(),
        MinCashWeight(0.05),
        MaxWeight(0.60),
        TurnoverLimit(0.25, constraint_type="soft", soft_weight=2.0),
    )
    .build()
)

policy = MPOPolicy(
    problem=problem,
    input_plan=InputPlan(expected_returns="forecast", max_horizons=10),
    min_history=180,
    name="custom_mean_variance_costs",
)

# %%
run = Allocation(
    market=market,
    policy=policy,
    forecasts=forecaster,
    initial_cash=100_000,
).at()

print(run.target_weights)

# %%
# The custom problem still returns the same PolicyRun diagnostics interface.
print(run.plan_metrics())

# %%
run.plot_weights()
run.projection.plot()
