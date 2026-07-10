# %%
import logging

import polars as pl

from qraft import (
    Allocation,
    AssetUniverse,
    Forecaster,
    HistoryWeighting,
    InputPlan,
    LogConfig,
    MarketData,
    MPOPolicy,
    PipelineConfig,
    SimulationForecastConfig,
    setup_logging,
)
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MaxWeight,
    MinCashWeight,
    TurnoverLimit,
)
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.INFO))


# %%
# Load the sample data used by the other notebooks.
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

assets = list(prices.columns[10:80])
universe = AssetUniverse(assets=assets, factors=list(factor_cols)[:4])
prices = prices.select("date", *universe.all_tickers)

market = MarketData.from_prices(
    prices,
    universe,
    cash=cash,
    history_weighting=HistoryWeighting("state_smooth", half_life=60),
)


# %%
# Minimal forecaster, MPO policy, and input plan.
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(horizon=10, method="bootstrap", n_sims=10_000),
    # refit_every=int(prices.height / 4),
    refit_every=1,
    seed=10,
)

policy = MPOPolicy.preset(
    "cvar_auto",
    risk_aversion=5.0,
    # constraints=(LongOnly(), FullyInvested()),
    constraints=(
        LongOnly(),
        FullyInvested(constraint_type="soft", soft_weight=1.0),
        MinCashWeight(limit=0.2),
        TurnoverLimit(limit=0.15, constraint_type="soft", soft_weight=2.0),
        MaxWeight(limit=0.15),
    ),
    min_history=252,
)

plan = InputPlan(expected_returns="forecast", risk="both", max_horizons=10)

# %%
# Allocation.at() returns the policy result object.
run = Allocation(market, policy, source=forecaster, plan=plan).at()

run.target_weights


# %%
run.plan_metrics()


# %%
run.terminal_cvar(alpha=0.05)


# %%
run.in_model_cvar(alpha=0.05)


# %%
run.plot_weights()


# %%
run.projection.plot()
