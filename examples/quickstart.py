# %%
import logging

import numpy as np

from qraft import (
    Allocation,
    Backtest,
    BacktestConfig,
    Forecaster,
    MeanView,
    MPOPolicy,
    PipelineConfig,
    RankingView,
    RebalanceSchedule,
    SimulationForecastConfig,
    ViewWindow,
    Views,
)
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MaxWeight,
    MinCashWeight,
    TurnoverLimit,
)
from qraft.construction.optimization import InputPlan
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_vix_market

setup_logging(LogConfig(level=logging.INFO))

# %%
# Start with market data. The synthetic example has five made-up tradable assets
# and three made-up non-tradable factors.
market_without_views = synthetic_vix_market()

# %%
# Add one latest-date view. Views operate on simple-return scenario panels, so
# the STRESS_INDEX threshold is a historical factor move rather than a raw level.
as_of = market_without_views.trading_bars[-1]
returns = market_without_views.returns_through(as_of)
high_stress_move = float(
    np.quantile(returns.values.get_column("STRESS_INDEX").to_numpy(), 0.80)
)

latest_view = ViewWindow(
    start=as_of,
    end=as_of,
    views=Views(
        [
            MeanView("STRESS_INDEX", ">=", high_stress_move),
            RankingView(["ORION", "TERRA", "CYPHER", "LUMEN", "NOVA"]),
        ],
        confidence=0.75,
    ),
    name="latest_risk_off",
)

market = market_without_views.with_views([latest_view])
view_report = market.view_report(as_of)

print(view_report.diagnostics)
view_report.plot(prob_mode="regular")

# %%
# Configure forecasts. Forecaster selects recipes through history, then
# simulates future paths from the selected recipe and scenario probabilities.
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, n_sims=500, method="bootstrap"),
    new_recipe_every=80,
    new_recipe_cadence="every_bar",
    seed=42,
)

forecast_run = forecaster.run(
    market,
    min_history=60,
    forecast_cadence="quarter_end",
)

latest_forecast = forecast_run.steps[-1].forecast
print(forecast_run.model_health_df())
latest_forecast.plot_asset_paths(subset="tradable", max_assets=5, ncols=3)

# %%
# Use a preset multi-period optimization policy. Only the first horizon is
# actionable; later horizons are useful for diagnostics and risk.
policy = MPOPolicy.preset(
    objective_type="mean_covariance",
    risk_aversion=0.5,
    constraints=(
        LongOnly(),
        FullyInvested(constraint_type="soft", soft_weight=10.0),
        MinCashWeight(0.05),
        MaxWeight(0.60),
        TurnoverLimit(0.80),
    ),
    input_plan=InputPlan(expected_returns="forecast", max_horizons=10),
    min_history=60,
    name="quickstart_mpo",
)

run = Allocation(
    market=market,
    policy=policy,
    forecasts=forecast_run,
    initial_cash=100_000,
).at(forecast_run.steps[-1].as_of)

print(run.target_weights)
print(run.plan_metrics())
run.plot_weights()
run.projection.plot()

# %%
# Risk is computed from the policy projection and the same forecast paths.
risk = run.risk()

cvar_contrib = risk.risk_contribution("cvar", alpha=0.05)
effective_bets = risk.effective_bets()

print(risk.summary_df(alpha=0.05))
cvar_contrib.plot()
effective_bets.plot()

# %%
# Backtest the same policy through history. This uses the rebalance schedule to
# make decisions on month-end bars and execute on the next trading bar.
backtest = Backtest(
    market=market,
    policy=policy,
    forecasts=forecaster,
    config=BacktestConfig(
        schedule=RebalanceSchedule("quarter_end"),
        initial_cash=100_000,
    ),
).run()

print(backtest.summary_df())
backtest.plot_nav()
backtest.plot_weights()
backtest.plot_drawdown()
backtest.plot_turnover_and_costs()
