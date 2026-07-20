# %%
import logging

from qraft import (
    Backtest,
    BacktestConfig,
    EqualWeightPolicy,
    Forecaster,
    MPOPolicy,
    PipelineConfig,
    RebalanceSchedule,
    SimulationForecastConfig,
)
from qraft.backtest.result.result import (
    plot_comparison,
    plot_nav,
    plot_returns_hist,
    plot_rolling_metrics,
)
from qraft.construction import LongOnly, MinCashWeight, TurnoverLimit
from qraft.construction.optimization import InputPlan
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_market_example

setup_logging(LogConfig(level=logging.INFO))

# %%
# Build an equal-weight benchmark and a forecast-driven MPO strategy so the
# example can show both single-result and comparison APIs.
market = synthetic_market_example()
config = BacktestConfig(
    schedule=RebalanceSchedule("month_end"),
    initial_cash=100,
)

balanced = Backtest(
    market=market,
    policy=EqualWeightPolicy(target_cash_weight=0.05, name="equal_weight_5_cash"),
    config=config,
).run()

forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=5, n_sims=1_000, method="bootstrap"),
    new_recipe_every=120,
    new_recipe_cadence="every_bar",
    seed=42,
)

mpo_policy = MPOPolicy.preset(
    objective_type="mean_covariance",
    risk_aversion=0.2,
    constraints=(LongOnly(), MinCashWeight(0.05), TurnoverLimit(0.30)),
    input_plan=InputPlan(expected_returns="forecast", max_horizons=5),
    min_history=20,
    name="mean_covariance_mpo",
)

mpo = Backtest(
    market=market,
    policy=mpo_policy,
    forecasts=forecaster,
    config=config,
).run()

# %%
# Summary metrics are available as a dataclass or a one-row Polars DataFrame.
print(mpo.summary_df())

# %%
# Windowing returns another BacktestResult over an inclusive date range. This is
# useful for looking at stress periods, recent performance, or validation folds.
midpoint = len(mpo.nav_dates) // 2
recent = mpo.window(mpo.nav_dates[midpoint], mpo.nav_dates[-1])
print(recent.summary_df())

# %%
# If the market has active scenario views, view_activity_df summarizes which
# rebalances used them and reports entropy/constraint diagnostics.
print(mpo.view_activity_df())

# %%
# Method-style plots cover the common single-result views.
mpo.plot_nav()
mpo.plot_weights()
mpo.plot_drawdown()
mpo.plot_turnover_and_costs()

# %%
# Module-level plotting helpers add distribution, rolling, and comparison views.
plot_returns_hist(mpo)
plot_rolling_metrics(mpo, window=60)
plot_nav([balanced, mpo], labels=["Equal weight", "MPO"])
plot_comparison([balanced, mpo], labels=["Equal weight", "MPO"])
