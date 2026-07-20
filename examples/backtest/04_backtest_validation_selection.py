# %%
import logging

from qraft import (
    BacktestConfig,
    Forecaster,
    MPOPolicy,
    PipelineConfig,
    RebalanceSchedule,
    SimulationForecastConfig,
    Validation,
    WalkForwardConfig,
)
from qraft.construction import LongOnly, MinCashWeight, TurnoverLimit
from qraft.construction.optimization import InputPlan
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_market_example

setup_logging(LogConfig(level=logging.INFO))

# %%
# Validation evaluates a grid of immutable policy variants, then reports which
# candidates held up out of sample across folds.
market = synthetic_market_example()

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=5, n_sims=750, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=7,
)

base_policy = MPOPolicy.preset(
    objective_type="mean_covariance",
    risk_aversion=None,
    constraints=(LongOnly(), MinCashWeight(0.05), TurnoverLimit(0.30)),
    input_plan=InputPlan(expected_returns="forecast", max_horizons=5),
    min_history=120,
    name="mean_variance_grid",
)

validation = Validation(
    market=market,
    base_policy=base_policy,
    grid={"risk_aversion": [2.0, 4.0, 8.0]},
    forecasts=forecaster,
    backtest_config=BacktestConfig(
        schedule=RebalanceSchedule("month_end"),
        initial_cash=100,
    ),
)

# %%
# Walk-forward validation trains/selects on one sequence of decision dates and
# scores the selected candidate on the next sequence.
walk_cfg = WalkForwardConfig(
    train_size=8,
    test_size=3,
    fold_step=3,
    metric="sharpe",
    max_held_fraction=0.6,
)
walk = validation.walk_forward(walk_cfg)

print(walk.summary_df())
print(walk.selected_params_df())

# %%
# Tune returns a selected policy and exposes its full-candidate backtest result.
tuned = validation.tune(walk, cfg=walk_cfg)
print(tuned.backtest.summary_df())

# %%
# Reports and selected backtests include plotting helpers for review notebooks.
walk.plot()
tuned.backtest.plot_nav()
tuned.backtest.plot_drawdown()
