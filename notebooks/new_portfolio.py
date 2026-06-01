# %%
import logging

import numpy as np
from construction.forecast_2 import PortfolioExecution
from construction.optimization.constraints import (
    FullyInvested,
    LongOnly,
    PortfolioConstraint,
    TurnoverLimit,
)
from construction.optimization.moments import HorizonMoments
from construction.policies import MPOPolicy
from construction.state import PortfolioState
from forecast.pipelines.forecasting import AssetUniverse, run_n_steps_forecast
from forecast.probability.distributions import state_smooth_probs
from forecast.scenarios.panel import ScenarioPanel
from policy import LogConfig
from utils.log import setup_logging
from utils.tiingo import import_tickers_and_factors

setup_logging(LogConfig(level=logging.WARNING))

# %%
# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

min_price = 15

cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= np.log(15)  # type: ignore[arg-type]
    )
]

data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:60])
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
step = 10
forecast_moms = HorizonMoments.from_forecast_paths(
    forecasts, horizons=step, expectation_tolerance=0.1, cash_path="data/cash.csv"
)

assets = list(forecast_moms.assets)

# %%
# %%
rng = np.random.default_rng(seed=1)
rand_shares = rng.integers(low=10, high=65, size=len(forecast_moms.assets))
# %%
state = PortfolioState.from_forecast_and_assets(
    asset_forecasts=forecasts, assets=assets, shares=rand_shares, cash=100_000
)
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(),
    # MaxWeight(limit=0.09),
    # MaxWeightTopN(top_n=10, sum_limit=0.4, constraint_type="soft", soft_weight=500),
    TurnoverLimit(limit=0.80),
]

# %%
cvar_policy = MPOPolicy(name="cvar_cuts", risk_aversion=0.2, constraints=constraints)
cvar_dec = cvar_policy.decide(state=state, moments=forecast_moms)

mc_policy = MPOPolicy(
    name="mean_covariance", risk_aversion=0.3, constraints=constraints
)
mc_dec = mc_policy.decide(state=state, moments=forecast_moms)


# %%

PortfolioExecution.from_policy_and_forecasts(
    policy_decision=cvar_dec, forecasts=forecasts, state=state, assets=assets
).plot("cum_performance")

PortfolioExecution.from_policy_and_forecasts(
    policy_decision=mc_dec, forecasts=forecasts, state=state, assets=assets
).plot("cum_performance")
