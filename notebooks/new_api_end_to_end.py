# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    Backtest,
    BacktestConfig,
    CMAConfig,
    Forecaster,
    HistoryWeighting,
    InputPlan,
    LogConfig,
    MarketData,
    MPOPolicy,
    PipelineConfig,
    PortfolioRisk,
    SimulationForecastConfig,
    Validation,
    Views,
    run_policy,
    setup_logging,
)
from qraft.backtest.configs import WalkForwardConfig
from qraft.construction import FullyInvested, LongOnly, MinCashWeight
from qraft.construction.optimization.inputs import PolicyInputs
from qraft.construction.state import PortfolioState
from qraft.core.scenarios.view_types import RankingView
from qraft.core.schedule import RebalanceSchedule
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# %%
# Data and causal market setup.
prices, factor_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

min_price = 15
cols_to_keep = [
    col
    for col in prices.columns
    if col == "date"
    or (
        prices[col].null_count() == 0
        and prices[col].dtype.is_numeric()
        and float(prices[col].min()) >= np.log(min_price)  # type: ignore[arg-type]
    )
]
prices = prices.select(cols_to_keep)

assets = list(prices.columns[10:25])
universe = AssetUniverse(assets=assets, factors=list(factor_cols)[:4])
prices = prices.select("date", *universe.all_tickers)

views = Views([RankingView(order=assets[:3])], confidence=0.35)
market = MarketData.from_log_prices(
    prices,
    universe,
    cash=cash,
    history_weighting=HistoryWeighting("state_smooth", half_life=45),
).with_view_events((prices["date"][-120], views))

# %%
# Forecast, input, and policy configuration.
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(
        horizon=6,
        method="cma",
        n_sims=1_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
    refit_every=max(12, int(prices.height / 4)),
    seed=7,
)
plan = InputPlan(expected_returns="forecast", risk="both", max_horizons=6)
backtest_config = BacktestConfig(schedule=RebalanceSchedule("quarter_end"))

base_policy = MPOPolicy.preset(
    objective_type="cvar_cuts",
    constraints=(
        LongOnly(),
        FullyInvested(constraint_type="soft", soft_weight=1.0),
        MinCashWeight(limit=0.10),
    ),
    min_history=252,
    name="cvar_template",
)

# %%
# Validate risk_aversion, then take the selected params/policy from ValidationResult.
validation_result = Validation(
    market=market,
    base_policy=base_policy,
    grid={"risk_aversion": [0.01, 0.03, 0.10]},
    source=forecaster,
    plan=plan,
    cv_config=WalkForwardConfig(
        train_size=4,
        test_size=1,
        fold_step=1,
        metric="sharpe",
    ),
    backtest_config=backtest_config,
).run()

selected_params = validation_result.selected_params
selected_policy = validation_result.selected_policy
selected_params.as_dict(), validation_result.report.summary_df

# %%
# Backtest the selected policy.
backtest_result = Backtest(
    market=market,
    policy=selected_policy,
    source=forecaster,
    plan=plan,
    config=backtest_config,
).run()

backtest_result.nav[-1]

# %%
# Risk report for the selected policy using the latest causal snapshot.
decision_bar = market.trading_bars[-2]
execution_bar = market.trading_bars[-1]
snapshot = market.snapshot_at(decision_bar, execution_bar)
forecast_paths = forecaster.forecast(snapshot.history, snapshot.universe)
policy_inputs = PolicyInputs.from_policy_sources(
    forecasts=forecast_paths,
    expected_returns=plan.expected_returns,
    risk="both",
    history=snapshot.history,
    max_horizons=plan.max_horizons,
    subset=plan.subset,
    pnl_type=plan.pnl_type,
    expectation_tolerance=plan.expectation_tolerance,
    mean_decay=plan.mean_decay,
    as_of=snapshot.t,
    cash_return=snapshot.cash_rate,
)
state = PortfolioState.from_cash(
    cash=backtest_config.initial_cash,
    assets=assets,
    asset_forecasts=forecast_paths,
)
policy_run = run_policy(
    selected_policy,
    state,
    forecast_paths,
    policy_inputs=policy_inputs,
)
if policy_run.projection is None:
    raise RuntimeError("Policy projection was not created.")

risk_report = PortfolioRisk.from_projection(policy_run.projection, forecast_paths)
risk_report.risk_contribution("cvar")
