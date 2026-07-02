# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    BacktestConfig,
    CMAConfig,
    Forecaster,
    HistoryWeighting,
    InputPlan,
    LogConfig,
    MarketData,
    MPOPolicy,
    PipelineConfig,
    SimulationForecastConfig,
    Validation,
    Views,
    setup_logging,
)
from qraft.backtest.configs import WalkForwardConfig
from qraft.construction import FullyInvested, LongOnly, MinCashWeight
from qraft.construction.policies.allocation import Allocation
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

min_price = 12
cols_to_keep = [
    col
    for col in prices.columns
    if col == "date"
    or (
        col != "DHIL"
        and prices[col].null_count() == 0
        and prices[col].dtype.is_numeric()
        and float(prices[col].min()) >= np.log(min_price)  # type: ignore[arg-type]
    )
]
prices = prices.select(cols_to_keep)

assets = list(prices.columns[10:90])
universe = AssetUniverse(assets=assets, factors=list(factor_cols)[:4])
prices = prices.select("date", *universe.all_tickers)

views = Views([RankingView(order=assets[:1])], confidence=0.35)
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
        horizon=10,
        method="cma",
        n_sims=10_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
    refit_every=max(12, int(prices.height / 4)),
    seed=7,
)
plan = InputPlan(expected_returns="forecast", risk="both")
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
# --- Phase 4: tune / tuned_policy --------------------------------------------------
# The same Validation object can tune with any metric — the grid is evaluated once.
val = Validation(
    market=market,
    base_policy=base_policy,
    grid={"risk_aversion": [0.05, 0.5, 1, 3, 5, 10]},
    source=forecaster,
    plan=plan,
    cv_config=WalkForwardConfig(),
    backtest_config=backtest_config,
)


# %%
# tuned_policy wraps tune() + apply_hyperparameters.
policy = val.tuned_policy(metric="sortino")

# %%
live = Allocation(market, policy, source=forecaster, plan=plan)
run = live.at()
risk = live.risk()

# %%
risk.effective_bets().plot()
