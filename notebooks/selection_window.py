# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    CMAConfig,
    Forecaster,
    Prior,
    LogConfig,
    MarketData,
    MPOPolicy,
    PipelineConfig,
    Validation,
    profile,
    setup_logging,
)
from qraft.backtest.configs import BacktestConfig, CombinatorialCVConfig, CVConfig
from qraft.backtest.selection import plot_combinatorial_report
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MinCashWeight,
    PortfolioConstraint,
    TurnoverLimit,
)
from qraft.construction.optimization.inputs import InputPlan
from qraft.core.configs import SimulationForecastConfig
from qraft.core.schedule import RebalanceSchedule
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# %%
# Data setup mirrors backtest_a.py, but keeps the universe small so a grid run is tractable.
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
data = data.with_columns(pl.selectors.numeric().exp())
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

data

# %%
min_price = 15
cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        col != "DHIL"
        and data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= min_price  # type: ignore[arg-type]
    )
]
data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:90])
universe = AssetUniverse(assets=tradable_assets, factors=list(factors_cols))
data = data.select("date", *universe.all_tickers)

market = MarketData.from_prices(
    data,
    universe,
    cash=cash,
    prior=Prior.time_conditioned(half_life=35),
)

# %%
# Base policy and the parameters to sweep.
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(constraint_type="soft", soft_weight=1.0),
    MinCashWeight(limit=0.25),
    TurnoverLimit(limit=0.15, constraint_type="soft", soft_weight=2.0),
]

base_policy = MPOPolicy.preset(
    objective_type="cvar_cuts",
    constraints=constraints,
    min_history=360,
)

risk_aversion_values = [*np.logspace(-2, 2, 9)]
# risk_aversion_values = [1, 5, 10, 15]
# risk_aversion_values = [1, 5]

grid = {
    "risk_aversion": risk_aversion_values
    # "transaction_cost_weight": [0.5, 1.0],
}

plan = InputPlan(
    expected_returns="forecast",
)
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(
        horizon=15,
        method="cma",
        n_sims=10_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
    refit_every=int(data.height / 3),
)

# %%
with profile():
    results = Validation(
        market=market,
        base_policy=base_policy,
        grid=grid,
        forecasts=forecaster,
        plan=plan,
        backtest_config=BacktestConfig(schedule=RebalanceSchedule("quarter_end")),
    ).combinatorial(
        CombinatorialCVConfig(
            cv_config=CVConfig(),
        )
    )


# %%
plot_combinatorial_report(results)
