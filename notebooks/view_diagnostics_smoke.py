# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    Backtest,
    BacktestConfig,
    Forecaster,
    InputPlan,
    LogConfig,
    MarketData,
    MPOPolicy,
    PipelineConfig,
    Prior,
    SimulationForecastConfig,
    Views,
    setup_logging,
)
from qraft.construction import FullyInvested, LongOnly, MinCashWeight, TurnoverLimit
from qraft.core.scenarios.view_types import MeanView, RankingView
from qraft.core.scenarios.views import ViewWindow
from qraft.core.schedule import RebalanceSchedule
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.INFO))

# %%
# Small sample universe for a fast, readable smoke test.
prices, factor_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
prices = prices.with_columns(pl.selectors.numeric().exp())
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

cols = [
    col
    for col in prices.columns
    if col == "date"
    or (
        col != "DHIL"
        and prices[col].null_count() == 0
        and prices[col].dtype.is_numeric()
        and float(prices[col].min()) >= 12  # type: ignore[arg-type]
    )
]
prices = prices.select(cols)

assets = list(prices.columns[10:14])
universe = AssetUniverse(assets=assets, factors=list(factor_cols)[:3])
prices = prices.select("date", *universe.all_tickers)

market0 = MarketData.from_prices(
    prices,
    universe,
    cash=cash,
    prior=Prior.time_conditioned(half_life=60),
)

# %%
# Create one active entropy-pooling view. Use returns_through(), not a private method.
view_date = prices["date"][-120]
view_end = prices["date"][-1]
view_asset = assets[0]
returns = market0.returns_through(view_date)
anchor = float(
    np.average(returns.values.get_column(view_asset).to_numpy(), weights=returns.prob)
)

market = market0.with_views(
    [
        ViewWindow(
            view_date,
            view_end,
            Views(
                [
                    MeanView(view_asset, ">=", anchor * 1.25),
                    RankingView(order=assets[:3]),
                ],
                confidence=0.75,
                solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
            ),
        )
    ]
)

# %%
# New MarketData API: no active view -> None; active EP view -> ViewedDistribution.
print("Before view is active:", market.view_report(prices["date"][20]))

# %%
report = market.view_report(view_date)
print("Active view diagnostics:")
print(report.diagnostics if report is not None else None)

# %%
# Existing ViewedDistribution tools still expose prior/posterior moments and plots.
print("Prior/posterior moments:")
print(
    pl.DataFrame(report.moments()).transpose(include_header=True)
    if report is not None
    else None
)

# %%
if report is not None:
    report.plot()

# %%
# Backtest path: diagnostics ride on OptimizerInputs -> BacktestPeriod -> BacktestResult.
plan = InputPlan(expected_returns="historical", max_horizons=1)
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=3, n_sims=200),
    refit_every=int(market.frame.height / 3),
    seed=10,
)
policy = MPOPolicy.preset(
    "cvar_auto",
    risk_aversion=2.0,
    constraints=(
        LongOnly(),
        FullyInvested(constraint_type="soft", soft_weight=1.0),
        MinCashWeight(limit=0.20, constraint_type="soft", soft_weight=2.0),
        TurnoverLimit(limit=0.15, constraint_type="soft", soft_weight=2.0),
    ),
    name="view-diagnostics-cvar",
    input_plan=plan,
    max_iter=80,
)
config = BacktestConfig(
    schedule=RebalanceSchedule("year_end"),
    initial_cash=100.0,
)

result = Backtest(
    market=market,
    policy=policy,
    forecasts=forecaster,
    config=config,
).run()

# %%
# New BacktestResult API: compact view activity table.
activity = result.view_activity_df()
print("View activity:")
print(activity)

# %%
print("Active view rows:")
print(activity.filter(pl.col("views_active")))

# %%
result.plot_nav()

# %%
result.plot_weights()
