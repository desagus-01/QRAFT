# %%
import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    EqualWeightPolicy,
    LogConfig,
    MPOPolicy,
    setup_logging,
)
from qraft.backtest.market import MarketData, WindowWeighting
from qraft.backtest.simulator import run_backtest
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MinCashWeight,
    PortfolioConstraint,
    TurnoverLimit,
)
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))

# %%
# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

min_price = 15

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

tradable_assets = list(data.columns[10:20])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)

# %%

mkt_dt = MarketData.from_log_prices(
    data, universe, cash=cash, weighting=WindowWeighting("state_smooth", half_life=126)
)

mkt_dt.history_through(t="2020-01-01")
# %%
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(constraint_type="soft", soft_weight=1.0),
    # FullyInvested(),
    MinCashWeight(limit=0.3, constraint_type="soft", soft_weight=1.0),
    # MaxWeight(limit=0.09),
    # MaxWeightTopN(top_n=10, sum_limit=0.4, constraint_type="soft", soft_weight=500),
    TurnoverLimit(limit=0.80),
]

policy = MPOPolicy.preset(
    objective_type="mean_covariance",
    risk_aversion=0.1,
    cash_path="data/cash.csv",
    constraints=constraints,
    expectation_tolerance=0.2,
)


# result = run_backtest(
#     mkt_dt,
#     policy,  # MPOPolicy -> requires_forecast=True
#     schedule=RebalanceSchedule("quarter_end"),
#     forecaster=PipelineForecaster(
#         simulation_config=SimulationForecastConfig(
#             method="cma",
#             n_sims=10_000,
#             cma_config=CMAConfig(target_copula="t"),
#         ),
#     ),
# )

result = run_backtest(market=mkt_dt, policy=EqualWeightPolicy)
# %%

# Build a Polars DataFrame

df = pl.DataFrame({"date": result.nav_dates, "nav": result.nav}).sort("date")

plt.figure(figsize=(12, 6))
plt.plot(df["date"].to_list(), df["nav"].to_list(), marker="o", linewidth=2)

plt.title("NAV Over Time")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
