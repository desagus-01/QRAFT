# %%
import logging

import numpy as np
import polars as pl

from qraft import (
    AssetUniverse,
    LogConfig,
    setup_logging,
)
from qraft.backtest.market import MarketData, WindowWeighting
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.INFO))

# %%
# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

min_price = 15

cash = pl.read_csv(
    "~/Documents/projects/fund/QRAFT/data/cash.csv", try_parse_dates=True
)

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

tradable_assets = list(data.columns[10:80])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)

# %%

mkt_dt = MarketData.from_log_prices(
    data, universe, cash=cash, weighting=WindowWeighting("state_smooth", half_life=126)
)

mkt_dt.history_through(t="2020-01-01")
