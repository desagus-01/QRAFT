# %%
"""Tour the public BacktestResult API, diagnostics, and plots."""

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
from qraft.utils.example_data import synthetic_vix_market

setup_logging(LogConfig(level=logging.INFO))

# %%
# Build an equal-weight benchmark and a forecast-driven MPO strategy so the
# example can show both single-result and comparison APIs.
market = synthetic_vix_market()
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
print(mpo.summary())
print(mpo.summary_df())
print(mpo.summary(risk_free_rate=0.02, active_only=False))

# %%
# Windowing returns another BacktestResult over an inclusive date range. This is
# useful for looking at stress periods, recent performance, or validation folds.
midpoint = len(mpo.nav_dates) // 2
recent = mpo.window(mpo.nav_dates[midpoint], mpo.nav_dates[-1])
print(recent.summary_df())

# %%
# Rebalance-level arrays are convenient for dashboards and custom diagnostics.
print(
    {
        "first_decisions": mpo.period_decision_bars[:3],
        "first_executions": mpo.period_execution_bars[:3],
        "period_returns": mpo.period_returns[:3],
        "period_turnovers": mpo.period_turnovers[:3],
        "period_costs": mpo.period_costs[:3],
        "period_cash": mpo.period_cash[:3],
        "total_transaction_cost": mpo.total_transaction_cost,
        "total_holding_cost": mpo.total_holding_cost,
        "total_cost": mpo.total_cost,
    }
)

print({"weights_shape": mpo.period_weights_array.shape})
print({"target_weights_shape": mpo.period_target_weights_array.shape})

# %%
# Diagnostics are attached both to the result and to each rebalance period. MPO
# periods include solver status and optimizer diagnostics.
print({"warnings": mpo.warnings_log})
print({"dropped_assets": mpo.dropped_assets})

for period in mpo.periods[:3]:
    print(
        {
            "decision_bar": period.decision_bar,
            "execution_bar": period.execution_bar,
            "solver_status": period.solver_status,
            "decision_error": period.decision_error,
            "cost": period.cost,
            "cash_weight_after": period.state_after.cash_weight,
            "dropped_assets": period.dropped_assets,
        }
    )

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
