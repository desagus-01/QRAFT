# %%
import logging

import numpy as np
from construction.optimization.constraints import (
    LongOnly,
    PortfolioConstraint,
    TurnoverLimit,
)
from construction.policies import MPOPolicy
from construction.policy_projection import PolicyProjection
from construction.state import PortfolioState
from forecast.config import LogConfig
from forecast.pipelines.forecasting import AssetUniverse, run_n_steps_forecast
from forecast.probability.distributions import state_smooth_probs
from forecast.scenarios.panel import ScenarioPanel
from risk.risk_report import PortfolioRisk
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
        and float(data[col].min()) >= np.log(min_price)  # type: ignore[arg-type]
    )
]

data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:30])
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
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    # FullyInvested(),
    # MaxWeight(limit=0.09),
    # MaxWeightTopN(top_n=10, sum_limit=0.4, constraint_type="soft", soft_weight=500),
    TurnoverLimit(limit=0.80),
]

policy = MPOPolicy.preset(
    objective_type="cvar_cuts",
    risk_aversion=0.2,
    cash_path="data/cash.csv",
    constraints=constraints,
    expectation_tolerance=0.1,
)
rng = np.random.default_rng(seed=1)
rand_shares = rng.integers(low=10, high=75, size=len(universe.assets))

state = PortfolioState.from_forecast_and_assets(
    asset_forecasts=forecasts,
    assets=universe.assets,
    shares=rand_shares,
    cash=100_000,
)
decision = policy.decide(state, forecasts)

# %%
s = PolicyProjection.from_decision(
    decision=decision,
    forecasts=forecasts,
    state=state,
)

s.plot(type="cum_performance")

# %%

x = PortfolioRisk.build(
    policy_projection=s,
    asset_forecasts=forecasts,
    original_data=historical_panel.to_frame(),
    auto_select_factors=True,
    criterion="bic",
    horizon=19,
)

x.effective_bets().plot()
