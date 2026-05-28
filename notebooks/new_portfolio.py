# %%
import logging

import cvxpy as cp
import numpy as np
from pipelines.forecasting import AssetUniverse, run_n_steps_forecast
from policy import LogConfig
from portfolio.forecast import portfolio_forecast
from portfolio.policy import LongOnly, TurnoverLimit
from portfolio.policy.constraints import MaxWeight, PortfolioConstraint
from portfolio.policy.moments import (
    HorizonMoments,
)
from portfolio.pre_built import multi_period_optimization
from probability.distributions import state_smooth_probs
from scenarios.panel import ScenarioPanel
from utils.log import setup_logging
from utils.tiingo import import_tickers_and_factors

setup_logging(LogConfig(level=logging.WARNING))

# %%
# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= 1  # type: ignore[arg-type]
    )
]

data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:20])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)

# %%
# ── Build historical ScenarioPanel ───────────────────────────────────
forecast_horizon = 20
n_sims = 30_000

prob_ex = state_smooth_probs(
    data.height,
    half_life=data.height / 2,
    time_based=True,
)

historical_panel = ScenarioPanel.from_frame(
    data,
    prob=prob_ex,
)


# %%
# ── Forecasting ──────────────────────────────────────────────────────
forecasts = run_n_steps_forecast(
    data=historical_panel.to_frame(),
    prob=historical_panel.prob,
    horizon=forecast_horizon,
    n_sims=n_sims,
    seed=3,
    universe=universe,
    method="cma",
    target_copula="t",
    back_to_price=True,
)


# %%
rebalancing_period = 10
forecast_moms = HorizonMoments.from_forecast_paths(
    forecasts,
    horizons=rebalancing_period,
    expectation_tolerance=0.1,
    cash_path="~/Documents/projects/fund/QRAFT/data/cash.csv",
)


# %%
assets = forecast_moms.assets
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    MaxWeight(limit=0.09),
    # MaxWeightTopN(top_n=10, sum_limit=0.4, constraint_type="soft", soft_weight=500),
    TurnoverLimit(limit=0.30),
]

n = len(assets)
initial_cash = 0.3


mpo_res = multi_period_optimization(
    objective_type="cvar_auto",
    horizons=rebalancing_period,
    n_assets=n,
    risk_aversion=0.1,
    moments=forecast_moms,
    current_weights=np.full(n, (1.0 - initial_cash) / n),
    current_cash=initial_cash,
    constraints=constraints,
    solver=cp.CLARABEL,
    # verbose=True,
)
# %%
opt_weights_cash = mpo_res.target_weights_by_asset_with_cash()

opt_weights_risk = mpo_res.target_weights_renormalized()
# %%

tradable_paths = forecasts.tradable_paths
portfolio_forecast(
    asset_forecasts=tradable_paths,
    path_probs=forecasts.path_probs,
    initial_asset_shares=opt_weights_risk,
)
