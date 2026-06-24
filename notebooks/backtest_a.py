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
    ScenarioPanel,
    setup_logging,
)
from qraft.backtest.inputs import ForecastInputsProvider
from qraft.backtest.market import MarketData, WindowWeighting
from qraft.backtest.schedule import RebalanceSchedule
from qraft.backtest.simulator import run_backtest
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MinCashWeight,
    PortfolioConstraint,
    TurnoverLimit,
)
from qraft.construction.optimization.moments import PolicyInputConfig
from qraft.core import state_smooth_probs
from qraft.core.configs import SimulationForecastConfig
from qraft.utils.backtest_viz import plot_backtest_dashboard
from qraft.utils.tiingo import import_tickers_and_factors

_fault_log = open("/tmp/backtest_fault.log", "w")
logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# setup_logging(LogConfig(log_file="backtest.log"))
# %%
print("kernel ready")

# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

min_price = 10

cash = pl.read_csv("data/cash.csv", try_parse_dates=True)


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
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)
prob_ex = state_smooth_probs(
    data.height,
    half_life=data.height / 2,
    time_based=True,
)


posterior_panel = ScenarioPanel.from_log_prices(
    data,
    prob=prob_ex,
)
# %%

mkt_dt = MarketData.from_log_prices(
    data, universe, cash=cash, weighting=WindowWeighting("state_smooth", half_life=60)
)

# %%
# forecasts = run_forecast(
#     panel=posterior_panel,
#     seed=3,
#     universe=universe,
#     include_fit_diagnostics=True,
#     simulation_config=SimulationForecastConfig(
#         method="cma", n_sims=10_000, cma_config=CMAConfig(target_copula="t")
#     ),
# )
#
#
# %%
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(constraint_type="soft", soft_weight=1.0),
    MinCashWeight(limit=0.25),
    TurnoverLimit(limit=0.10),
]

policy = MPOPolicy.preset(
    objective_type="cvar_cuts",
    risk_aversion=0.01,
    constraints=constraints,
    min_history=504,
)

forecast_provider = ForecastInputsProvider(
    input_config=PolicyInputConfig(
        cash_path="data/cash.csv",
        expected_returns="forecast",
        risk="both",
    ),
    simulation_config=SimulationForecastConfig(
        horizon=15, method="cma", n_sims=10_000, cma_config=CMAConfig(target_copula="t")
    ),
    pipeline_config=PipelineConfig(exclude_non_invariants=False),
    recipe_every=4,
    forecast_every=1,
)

result = run_backtest(
    mkt_dt,
    policy,
    inputs=forecast_provider,
    schedule=RebalanceSchedule("month_end"),
)

# %%

plot_backtest_dashboard(result)
