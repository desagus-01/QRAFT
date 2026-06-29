# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    CMAConfig,
    LogConfig,
    MPOPolicy,
    PipelineConfig,
    profile,
    setup_logging,
)
from qraft.backtest.inputs import ForecastInputsProvider
from qraft.backtest.market import MarketData, WindowWeighting
from qraft.backtest.schedule import RebalanceSchedule
from qraft.backtest.selection import combinatorial_purged
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MinCashWeight,
    PortfolioConstraint,
    TurnoverLimit,
)
from qraft.construction.optimization.moments import PolicyInputConfig
from qraft.core.configs import (
    BacktestConfig,
    CombinatorialCVConfig,
    CVConfig,
    ForecastProviderConfig,
    SimulationForecastConfig,
)
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# %%
# Data setup mirrors backtest_a.py, but keeps the universe small so a grid run is tractable.
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

min_price = 15
cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= np.log(min_price)  # type: ignore[arg-type]
    )
]
data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:90])
universe = AssetUniverse(assets=tradable_assets, factors=list(factors_cols))
data = data.select("date", *universe.all_tickers)

market = MarketData.from_log_prices(
    data,
    universe,
    cash=cash,
    weighting=WindowWeighting("state_smooth", half_life=35),
)

# %%
# Base policy and the parameters to sweep.
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(constraint_type="soft", soft_weight=1.0),
    MinCashWeight(limit=0.25),
    TurnoverLimit(limit=0.15, constraint_type="soft", soft_weight=1.0),
]

base_policy = MPOPolicy.preset(
    objective_type="cvar_cuts",
    risk_aversion=0.01,
    constraints=constraints,
    min_history=504,
)

risk_aversion_values = [0, *np.logspace(-2, 2, 9)]
risk_aversion_values = [2, 3, 4, 8]
risk_aversion_values = [1, 5, 10, 15]

grid = {
    "risk_aversion": risk_aversion_values
    # "transaction_cost_weight": [0.5, 1.0],
}

forecast_provider = ForecastInputsProvider(
    input_config=PolicyInputConfig(
        cash_path="data/cash.csv",
        expected_returns="forecast",
    ),
    policy=base_policy,
    simulation_config=SimulationForecastConfig(
        horizon=15,
        method="cma",
        n_sims=10_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
    pipeline_config=PipelineConfig(exclude_non_invariants=False),
    provider_config=ForecastProviderConfig(refit_every=3),
)

# %%
with profile():
    results = combinatorial_purged(
        market,
        base_policy,
        grid,
        forecast_provider,
        cv_config=CombinatorialCVConfig(
            cv_config=CVConfig(),
        ),
        backtest_config=BacktestConfig(schedule=RebalanceSchedule("quarter_end")),
    )


# %%
results.summary_df()
