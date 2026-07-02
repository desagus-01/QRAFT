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
    ScenarioPanel,
    SimulationForecastConfig,
    Validation,
    Views,
    run_policy,
    setup_logging,
)
from qraft.backtest.configs import WalkForwardConfig
from qraft.backtest.selection.candidates import apply_hyperparameters
from qraft.construction import FullyInvested, LongOnly, MinCashWeight
from qraft.construction.optimization.inputs import PolicyInputs
from qraft.construction.state import PortfolioState
from qraft.core.scenarios.view_types import MeanView, RankingView
from qraft.core.schedule import RebalanceSchedule
from qraft.utils.backtest_viz import plot_backtest_dashboard
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# %%
# Data: keep this intentionally small so the full workflow can be smoke-tested.
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
factors = list(factor_cols)[:4]
universe = AssetUniverse(assets=assets, factors=factors)
prices = prices.select("date", *universe.all_tickers)

market = MarketData.from_log_prices(
    prices,
    universe,
    cash=cash,
    history_weighting=HistoryWeighting("state_smooth", half_life=45),
)

# %%
# Views: current-only views are for live/latest analysis. MarketData will reject
# current-only views in Backtest/Validation to prevent leaking today's beliefs
# into historical decisions. Use with_view_events(...) for causal backtests.
subjective_views = Views(
    specs=[
        RankingView(order=assets[:3]),
        MeanView(asset=assets[0], sign=">=", target=0.0),
    ],
    confidence=0.35,
)
live_market = market.with_current_views(subjective_views)
viewed_panel: ScenarioPanel = live_market.current_history()

causal_market = market.with_view_events(
    (market.trading_bars[-120], subjective_views),
)

viewed_panel.values.head()

# %%
# Shared forecast/input configuration.
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

# %%
# Base policy deliberately leaves risk_aversion unset. Validation supplies it via grid.
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
grid = {"risk_aversion": [0.01, 0.03, 0.10]}

# %%
validation = Validation(
    market=causal_market,
    base_policy=base_policy,
    grid=grid,
    source=forecaster,
    plan=plan,
    cv_config=WalkForwardConfig(
        train_size=4,
        test_size=1,
        fold_step=1,
        risk_free_rate=0.0,
        metric="sharpe",
    ),
    backtest_config=backtest_config,
)
report = validation.run()

report.summary_df

# %%
# Pick the most recent walk-forward selection as the final policy.
selected_counts = report.selection_counts_df
selected_params = report.folds[-1].selection.selected_params
if selected_params is None:
    raise RuntimeError("Validation did not select a policy candidate.")

selected_policy = apply_hyperparameters(base_policy, selected_params)
selected_params.as_dict(), selected_counts

# %%
# Run the selected policy through the Backtest facade.
backtest_result = Backtest(
    market=causal_market,
    policy=selected_policy,
    source=forecaster,
    plan=plan,
    config=backtest_config,
).run()

plot_backtest_dashboard(backtest_result)

# %%
# Risk report from a policy projection. Use the latest decision snapshot and the
# same forecaster to create forward paths, then build one set of policy inputs.
decision_bar = market.trading_bars[-2]
execution_bar = market.trading_bars[-1]
snapshot = live_market.snapshot_at(decision_bar, execution_bar)
live_history = live_market.current_history()
forecast_paths = forecaster.forecast(live_history, snapshot.universe)
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

# For an explicit projection/risk report, run one policy decision against the forecast.
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

risk_report = PortfolioRisk.from_projection(
    policy_projection=policy_run.projection,
    forecasts=forecast_paths,
)
risk_report.risk_contribution("cvar")
