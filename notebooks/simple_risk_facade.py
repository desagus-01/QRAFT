# %%
import logging

import polars as pl

from qraft import (
    Allocation,
    AssetUniverse,
    CMAConfig,
    Forecaster,
    Prior,
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

assets = list(prices.columns[10:80])
universe = AssetUniverse(assets=assets, factors=list(factor_cols)[:4])
prices = prices.select("date", *universe.all_tickers)

market = MarketData.from_prices(
    prices,
    universe,
    cash=cash,
    prior=Prior.time_conditioned(half_life=60),
)


# %%
# Forecast and optimize once.
forecaster = Forecaster(
    pipeline=PipelineConfig(exclude_non_invariants=False),
    simulation=SimulationForecastConfig(
        horizon=10,
        method="cma",
        n_sims=10_000,
        cma_config=CMAConfig(target_copula="t"),
    ),
    refit_every=int(prices.height / 4),
    seed=10,
)
plan = InputPlan(expected_returns="forecast")

policy = MPOPolicy.preset(
    "cvar_auto",
    risk_aversion=2.0,
    constraints=(
        LongOnly(),
        FullyInvested(constraint_type="soft", soft_weight=10.0),
        MinCashWeight(limit=0.20),
        TurnoverLimit(limit=0.15, constraint_type="soft", soft_weight=2.0),
        MaxWeight(limit=0.2),
    ),
    min_history=252,
)

run = Allocation(market, policy, forecasts=forecaster, plan=plan).at()


# %%
# Policy diagnostics live directly on PolicyRun.
run.target_weights


# %%
print(run.plan_metrics())


# %%
print(run.terminal_cvar(alpha=0.05))


# %%
print(run.in_model_cvar(alpha=0.05))


# %%
print(run.plot_weights())


# %%
# Risk facade returns PortfolioRisk.
risk = run.risk()

print(risk.summary_df(alpha=0.05))


# %%
print(risk.risk_contribution("cvar", alpha=0.05))


# %%
risk.risk_contribution("cvar", alpha=0.05).plot()


# %%
risk.risk_contribution("var", alpha=0.05).plot()

# %%

risk.effective_bets().plot()

# %%
run.projection.plot()
