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
    setup_logging,
)
from qraft.backtest.inputs import ForecastInputsProvider
from qraft.backtest.market import MarketData, WindowWeighting
from qraft.backtest.schedule import RebalanceSchedule
from qraft.backtest.selection import run_selection_window
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MinCashWeight,
    PortfolioConstraint,
    TurnoverLimit,
)
from qraft.construction.optimization.moments import PolicyInputConfig
from qraft.core.configs import SimulationForecastConfig
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

tradable_assets = list(data.columns[10:20])
universe = AssetUniverse(assets=tradable_assets, factors=list(factors_cols))
data = data.select("date", *universe.all_tickers)

market = MarketData.from_log_prices(
    data,
    universe,
    cash=cash,
    weighting=WindowWeighting("state_smooth", half_life=60),
)

# %%
# Base policy and the parameters to sweep.
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(constraint_type="soft", soft_weight=1.0),
    MinCashWeight(limit=0.25),
    TurnoverLimit(limit=0.15),
]

base_policy = MPOPolicy.preset(
    objective_type="cvar_cuts",
    risk_aversion=0.01,
    constraints=constraints,
    min_history=504,
)

grid = {
    "risk_aversion": [0.005, 0.01, 0.02, 0.1, 0.2],
    # "transaction_cost_weight": [0.5, 1.0],
}

forecast_provider = ForecastInputsProvider(
    input_config=PolicyInputConfig(
        cash_path="data/cash.csv",
        expected_returns="forecast",
        risk="both",
    ),
    simulation_config=SimulationForecastConfig(
        horizon=15,
        method="cma",
        n_sims=10_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
    pipeline_config=PipelineConfig(exclude_non_invariants=False),
    recipe_every=4,
)

# %%
# Efficient path: this unites candidate expansion, one shared precompute pass,
# and evaluation of every candidate.
results = run_selection_window(
    market,
    base_policy,
    grid,
    forecast_provider,
    schedule=RebalanceSchedule("quarter_end"),
)

# %%
# Inspect candidate scores. Failures stay in the result list instead of aborting the sweep.
rows = []
for result in results:
    params = result.params.as_dict()
    if result.failure is not None:
        rows.append({**params, "status": "failed", "reason": result.failure.reason})
        continue

    assert result.summary is not None
    rows.append(
        {
            **params,
            "status": "ok",
            "sharpe": result.summary.sharpe,
            "annualised_return": result.summary.annualised_return,
            "max_drawdown": result.summary.max_drawdown,
            "held_fraction": result.summary.held_fraction,
            "n_solver_failures": result.summary.n_solver_failures,
        }
    )

scores = pl.DataFrame(rows).sort("sharpe", descending=True)
scores

# %%
best = next(result for result in results if result.failure is None)
best.params, best.summary
