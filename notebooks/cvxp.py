# %%
import logging

import cvxpy as cp
import numpy as np
from pipelines.forecasting import AssetUniverse, run_n_steps_forecast
from policy import LogConfig
from portfolio.policy import FullyInvested, LongOnly, TurnoverLimit
from portfolio.policy.constraints import MaxWeight, MaxWeightTopN, PortfolioConstraint
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

tradable_assets = list(data.columns[10:60])
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
    MaxWeight(limit=0.06),
    MaxWeightTopN(top_n=10, sum_limit=0.3),
    TurnoverLimit(limit=0.35),
]


x = multi_period_optimization(
    type="cvar_auto",
    horizons=h,
    n_assets=len(assets),
    risk_aversion=1.0,
    moments=forecast_moms,
    current_weights=np.full(len(assets), 1 / len(assets)),
    constraints=constraints,
    # verbose=True,
    solver=cp.CLARABEL,
)

# %%

x.target_weights_by_asset
# %%
w = np.array(list(x.target_weights_by_asset.values()))
w0 = np.full(len(w), 1 / len(w))

total_abs_drift = np.sum(np.abs(w - w0))
one_way_turnover = 0.5 * total_abs_drift
max_weight = w.max()
min_weight = w.min()

print("Total absolute drift:", total_abs_drift)
print("One-way turnover:", one_way_turnover)
print("Max weight:", max_weight)
print("Min weight:", min_weight)
