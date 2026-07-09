# %%
import logging

import polars as pl

from qraft import (
    AssetUniverse,
    Backtest,
    BacktestConfig,
    CMAConfig,
    Forecaster,
    HistoryWeighting,
    LogConfig,
    MarketData,
    MPOPolicy,
    PipelineConfig,
    ScenarioPanel,
    setup_logging,
)
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MinCashWeight,
    PortfolioConstraint,
    TurnoverLimit,
)
from qraft.construction.optimization.inputs import InputPlan
from qraft.core import state_smooth_probs
from qraft.core.configs import SimulationForecastConfig
from qraft.core.schedule import RebalanceSchedule
from qraft.backtest.result.render import plot_backtest_dashboard
from qraft.utils.tiingo import import_tickers_and_factors

_fault_log = open("/tmp/backtest_fault.log", "w")
logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# setup_logging(LogConfig(log_file="backtest.log"))
# %%

# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
data = data.with_columns(pl.selectors.numeric().exp())

min_price = 15

cash = pl.read_csv("data/cash.csv", try_parse_dates=True)


cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= min_price  # type: ignore[arg-type]
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


posterior_panel = ScenarioPanel.from_prices(
    data,
    prob=prob_ex,
)
# %%

mkt_dt = MarketData.from_prices(
    data,
    universe,
    cash=cash,
    history_weighting=HistoryWeighting("state_smooth", half_life=60),
)

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
    min_history=250,
)

plan = InputPlan(
    expected_returns="forecast",
    risk="both",
)
schedule = RebalanceSchedule("year_end")

result = Backtest(
    market=mkt_dt,
    policy=policy,
    source=Forecaster(
        simulation=SimulationForecastConfig(
            horizon=15,
            method="cma",
            n_sims=20_000,
            cma_config=CMAConfig(target_copula="t"),
        ),
        pipeline=PipelineConfig(exclude_non_invariants=False),
        refit_every=int(data.height / 3),
    ),
    plan=plan,
    config=BacktestConfig(schedule=schedule),
).run()

# %%

plot_backtest_dashboard(result)
