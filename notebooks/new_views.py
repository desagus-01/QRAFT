# %%
import logging

import polars as pl

from qraft import (
    AssetUniverse,
    HistoryWeighting,
    LogConfig,
    MarketData,
    Views,
    setup_logging,
)
from qraft.core.scenarios.view_types import RankingView
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.INFO))

# %%
# Data and causal market setup.
prices, factor_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)
prices = prices.with_columns(pl.selectors.numeric().exp())
cash = pl.read_csv("data/cash.csv", try_parse_dates=True)

min_price = 12
cols_to_keep = [
    col
    for col in prices.columns
    if col == "date"
    or (
        col != "DHIL"
        and prices[col].null_count() == 0
        and prices[col].dtype.is_numeric()
        and float(prices[col].min()) >= min_price  # type: ignore[arg-type]
    )
]
prices = prices.select(cols_to_keep)

assets = list(prices.columns[10:90])
universe = AssetUniverse(assets=assets, factors=list(factor_cols)[:4])
prices = prices.select("date", *universe.all_tickers)
# %%
views = Views([RankingView(order=["CUBE", "MA", "CVCO"])], confidence=0.35)
market = MarketData.from_prices(
    prices,
    universe,
    cash=cash,
    history_weighting=HistoryWeighting("state_smooth", half_life=60),
).with_view_events((prices["date"][-120], views))

# %%
