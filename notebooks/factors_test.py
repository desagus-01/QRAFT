# %%
import logging

import polars as pl
from forecast.config import LogConfig
from pipelines.forecasting import AssetUniverse, run_n_steps_forecast
from portfolio import (
    build_equal_weight_portfolio_from_df,
    equal_weight_target_weights,
    portfolio_forecast,
)
from portfolio.attribution.performance import portfolio_factor_attribution
from portfolio.attribution.risk import (
    PortfolioRiskAttribution,
)
from portfolio.risk import cvar, var
from probability.distributions import state_smooth_probs
from scenarios.panel import ScenarioPanel
from utils.helpers import wide_to_long
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
    if col == "date" or (data[col].null_count() == 0 and data[col].min() >= 1)
]

data = data.select(cols_to_keep)

tradable_assets = list(data.columns[15:20])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)

# %%
# ── Build historical ScenarioPanel ───────────────────────────────────
#
# This is the realised/historical weighted panel.
# It is not a forecast object.
# It gives the forecasting pipeline clean input data + probabilities.

horizon = 30
n_sims = 30_000
analysis_horizon = 30

prob_ex = state_smooth_probs(
    data.height,
    half_life=120,
    time_based=True,
)

historical_panel = ScenarioPanel.from_frame(
    data,
    prob=prob_ex,
)

historical_panel

# %%
# ── Portfolio construction ───────────────────────────────────────────
#
# This still uses realised data.
# Keep it separate from forecast paths.

long = wide_to_long(
    historical_panel.to_frame(),
    assets=universe.all_tickers,
)

d2 = long.with_columns(pl.col("adj_close").exp().alias("adj_close")).filter(
    pl.col("ticker").is_in(tradable_assets)
)

port = build_equal_weight_portfolio_from_df(
    d2,
    initial_value=10_000,
)

# %%
# ── Forecasting ──────────────────────────────────────────────────────
#
# run_n_steps_forecast remains the orchestration function.
# ScenarioPanel does not own the forecasting pipeline.

forecasts = run_n_steps_forecast(
    data=historical_panel.to_frame(),
    prob=historical_panel.prob,
    horizon=horizon,
    n_sims=n_sims,
    seed=2,
    universe=universe,
    method="cma",
    target_copula="t",
    back_to_price=True,
)

forecasts

# %%
# ── Horizon slices as ScenarioPanel ──────────────────────────────────
#
# These are now the clean bridge between Monte Carlo paths and
# downstream scenario/risk modelling.
#
# rows    = simulated paths
# columns = assets / factors
# prob    = path probabilities

forecast_panel_h = forecasts.at_step(
    analysis_horizon,
    subset="all",
)

tradable_panel_h = forecasts.at_step(
    analysis_horizon,
    subset="tradable",
)

factor_panel_h = forecasts.at_step(
    analysis_horizon,
    subset="factors",
)

forecast_panel_h.values.head()

# %%
# ── Portfolio forecast ───────────────────────────────────────────────
#
# Portfolio forecast still needs the full path object, not one horizon.
# So we continue using forecasts.tradable_paths here.

tradable_paths = forecasts.tradable_paths

target_weights = equal_weight_target_weights(list(tradable_paths.keys()))

port_forecast = portfolio_forecast(
    asset_forecasts=tradable_paths,
    path_probs=forecasts.path_probs,
    initial_asset_shares=port.shares_mapping,
    initial_prices=port.t0_prices,
    pnl_type="relative",
    weight_mode="static",
    target_weights=target_weights,
)

port_forecast.plot(
    end_horizon=analysis_horizon,
    plot_cumulative=True,
)
# %%
# ── Factor attribution at selected horizon ───────────────────────────

perf_attr = portfolio_factor_attribution(
    portfolio_forecast=port_forecast,
    factors_forecast=forecasts.factor_paths,
    original_data=historical_panel.to_frame(),
    horizon=analysis_horizon,
    auto_select_factors=True,
    criterion="bic",
)

perf_attr.joint_distribution.head()

# %%
# ── Portfolio loss distribution ──────────────────────────────────────

portfolio_h30 = port_forecast.at_period(30)
losses_h30 = port_forecast.loss_at_period(30)

var_h30 = var(
    distribution=portfolio_h30.values["loss"].to_numpy(),
    prob=portfolio_h30.prob,
    alpha=0.05,
    method="empirical",
)

cvar_h30 = cvar(
    distribution=portfolio_h30.values["loss"].to_numpy(),
    prob=portfolio_h30.prob,
    alpha=0.05,
    method="empirical",
)

var_h30, cvar_h30
# %%
# ── Risk attribution ─────────────────────────────────────────────────
risk_attr = PortfolioRiskAttribution.from_performance_attribution(perf_attr)
risk_attr.cvar(alpha=0.05)
