# %%
import logging

import cvxpy as cp
import numpy as np
from pipelines.forecasting import AssetUniverse, run_n_steps_forecast
from policy import LogConfig
from portfolio.policy import FullyInvested, LongOnly, MinWeight, TurnoverLimit
from portfolio.policy.constraints import MaxWeight, PortfolioConstraint
from portfolio.policy.moments import (
    HorizonMoments,
)
from portfolio.pre_built import classic_mpo, cvar_mpo_cuts
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

tradable_assets = list(data.columns[10:90])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)

# %%
# ── Build historical ScenarioPanel ───────────────────────────────────
horizon = 30
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

historical_panel

# %%
# ── Forecasting ──────────────────────────────────────────────────────
forecasts = run_n_steps_forecast(
    data=historical_panel.to_frame(),
    prob=historical_panel.prob,
    horizon=horizon,
    n_sims=n_sims,
    seed=3,
    universe=universe,
    method="cma",
    target_copula="t",
    back_to_price=True,
)

# %%
h = 10
forecast_moms = HorizonMoments.from_forecast_paths(
    forecasts, horizons=h, expectation_tolerance=1.0
)

assets = forecast_moms.assets
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(),
    MaxWeight(limit=0.13),
    MinWeight(limit=0.01),
    TurnoverLimit(limit=0.1),
]


x = classic_mpo(
    horizons=h,
    n_assets=len(assets),
    risk_aversion=1.0,
    moments=forecast_moms,
    current_weights=np.full(len(assets), 1 / len(assets)),
    constraints=constraints,
    # verbose=True,
    solver=cp.CLARABEL,
)

c = cvar_mpo_cuts(
    horizons=h,
    n_assets=len(assets),
    cvar_aversion=1,
    moments=forecast_moms,
    current_weights=np.full(len(assets), 1 / len(assets)),
    constraints=constraints,
    # verbose=True,
    solver=cp.CLARABEL,
)
