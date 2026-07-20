# %%
import logging

from qraft import (
    Backtest,
    BacktestConfig,
    Forecaster,
    FullyInvested,
    MPOPolicy,
    PipelineConfig,
    RebalanceSchedule,
    SimulationForecastConfig,
    TurnoverLimit,
)
from qraft.construction import LongOnly, MinCashWeight
from qraft.construction.optimization import InputPlan
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_vix_market

setup_logging(LogConfig(level=logging.INFO))

# %%
# Build market data and configure forecasts. Backtest will precompute optimizer
# inputs on the rebalance dates before running the simulation loop.
market = synthetic_vix_market()

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, n_sims=10_000, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

# %%
# The input plan tells QRAFT how forecast paths become optimizer inputs. This
# policy uses forecast expected returns and covariance inputs over 10 horizons.
plan = InputPlan(expected_returns="forecast", max_horizons=10)

policy = MPOPolicy.preset(
    objective_type="cvar_auto",
    risk_aversion=5.0,
    constraints=(
        LongOnly(),
        FullyInvested(constraint_type="soft"),
        MinCashWeight(0.05),
        TurnoverLimit(0.25),
    ),
    input_plan=plan,
    min_history=30,
    name="cvar_auto_backtest",
)

config = BacktestConfig(
    schedule=RebalanceSchedule("quarter_end"),
    initial_cash=100,
)

# %%
# Forecasts are passed directly to Backtest. The engine calls the policy at each
# decision date, executes on the configured execution date, and records results.
result = Backtest(
    market=market,
    policy=policy,
    forecasts=forecaster,
    config=config,
).run()

# %%
print(result.summary_df())

# %%
result.plot_nav()
result.plot_weights()
result.plot_drawdown()
result.plot_turnover_and_costs()
