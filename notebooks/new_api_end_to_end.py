# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    BacktestConfig,
    CMAConfig,
    CombinatorialCVConfig,
    Forecaster,
    InputPlan,
    LogConfig,
    MarketData,
    MPOPolicy,
    PipelineConfig,
    Prior,
    SimulationForecastConfig,
    Validation,
    Views,
    setup_logging,
)
from qraft.construction import FullyInvested, LongOnly, MinCashWeight, TurnoverLimit
from qraft.construction.policies.allocation import Allocation
from qraft.core.scenarios.view_types import MeanView, QuantileView, RankingView, StdView
from qraft.core.schedule import RebalanceSchedule
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.INFO))

# %%
# Data and causal market setup.
prices, factor_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
prices = prices.with_columns(pl.selectors.numeric().exp())
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

min_price = 12
cols_to_keep = [
    col
    for col in prices.columns
    if col == "date"
    or (
        col != "DHIL"
        and prices[col].null_count() == 0
        and prices[col].dtype.is_numeric()
        and float(prices[col].min()) >= min_price  # type: ignore[arg-type]
    )
]
prices = prices.select(cols_to_keep)

assets = list(prices.columns[10:30])
universe = AssetUniverse(assets=assets, factors=list(factor_cols)[:4])
prices = prices.select("date", *universe.all_tickers)

market_without_views = MarketData.from_prices(
    prices,
    universe,
    cash=cash,
    prior=Prior.time_conditioned(half_life=60),
)

view_asset = assets[0]
view_dates = [prices["date"][-360], prices["date"][-240], prices["date"][-120]]
date_list = prices["date"].to_list()
view_ends = [
    date_list[date_list.index(view_dates[1]) - 1],
    date_list[date_list.index(view_dates[2]) - 1],
    prices["date"][-1],
]
return_panels = [
    market_without_views._simple_return_history_through(view_date)
    for view_date in view_dates
]
normal_averages = [
    np.average(panel.values.get_column(view_asset).to_numpy(), weights=panel.prob)
    for panel in return_panels
]

view_events = [
    (
        view_dates[0],
        view_ends[0],
        Views(
            [
                MeanView(view_asset, ">=", normal_averages[0] * 1.25),
                RankingView(order=assets[:3]),
            ],
            confidence=0.55,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
    (
        view_dates[1],
        view_ends[1],
        Views(
            [
                MeanView(assets[1], "<=", normal_averages[1] * 0.75),
                RankingView(order=[assets[2], assets[0], assets[1]]),
            ],
            confidence=0.75,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
    (
        view_dates[2],
        view_ends[2],
        Views(
            [
                MeanView(view_asset, "==", normal_averages[2] * 1.10),
                StdView(view_asset, "<=", 0.035),
                QuantileView(view_asset, 0.25, 0.20),
            ],
            confidence=0.90,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
]
market = market_without_views.with_views(*view_events)


# %%
# Forecast, input, and policy configuration.
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(
        horizon=10,
        method="cma",
        n_sims=1_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
    refit_every=int(prices.height / 4),
    seed=10,
)
plan = InputPlan(expected_returns="forecast")
backtest_config = BacktestConfig(schedule=RebalanceSchedule("quarter_end"))

base_policy = MPOPolicy.preset(
    objective_type="cvar_cuts",
    constraints=(
        LongOnly(),
        FullyInvested(constraint_type="soft", soft_weight=1.0),
        MinCashWeight(limit=0.20, constraint_type="soft", soft_weight=2.0),
        TurnoverLimit(limit=0.15, constraint_type="soft", soft_weight=2.0),
    ),
    min_history=252,
    name="cvar_template",
    input_plan=plan,
)

# %%
# --- Phase 4: tune --------------------------------------------------
# The same Validation object can tune with any metric — the grid is evaluated once.
val = Validation(
    market=market,
    base_policy=base_policy,
    grid={"risk_aversion": [2, 3, 4, 5]},
    forecasts=forecaster,
    backtest_config=backtest_config,
)

# %%
# %%
cv_config = CombinatorialCVConfig(n_groups=4, n_test_groups=1)
report = val.combinatorial(cv_config)
# %%
tuned = val.tune(report, cfg=cv_config, score="sortino")
# tuned = val.tune(report, score="sharpe")
policy = tuned.selected_policy
tuned.selected_params

# %%
live = Allocation(market, policy, forecasts=forecaster)
run = live.at()
risk = run.risk()

run.projection.plot()
# %%
risk.effective_bets().plot()
# %%
