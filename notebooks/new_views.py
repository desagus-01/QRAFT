# %%
import logging

import polars as pl

from qraft import (
    AssetUniverse,
    Prior,
    LogConfig,
    MarketData,
    Views,
    setup_logging,
)
from qraft.core.scenarios.view_types import MeanView, QuantileView, RankingView, StdView
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
view_asset = "CUBE"
view_dates = [prices["date"][-360], prices["date"][-240], prices["date"][-120]]
market_without_views = MarketData.from_prices(
    prices,
    universe,
    cash=cash,
    prior=Prior.time_conditioned(half_life=60),
)
normal_averages = [
    market_without_views.viewed_returns(Views([]), t=view_date).moments()[view_asset][
        "prior_mean"
    ]
    for view_date in view_dates
]

view_events = [
    (
        view_dates[0],
        Views(
            [
                MeanView("CUBE", ">=", normal_averages[0] * 1.25),
                RankingView(order=["CUBE", "MA", "CVCO"]),
            ],
            confidence=0.55,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
    (
        view_dates[1],
        Views(
            [
                MeanView("MA", "<=", normal_averages[1] * 0.75),
                RankingView(order=["CVCO", "CUBE", "MA"]),
            ],
            confidence=0.75,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
    (
        view_dates[2],
        Views(
            [
                MeanView("CUBE", "==", normal_averages[2] * 1.10),
                StdView("CUBE", "<=", 0.035),
                QuantileView("CUBE", 0.25, 0.20),
            ],
            confidence=0.90,
            solver_kwargs={"eps": 1e-7, "max_iters": 50_000},
        ),
    ),
]
market = MarketData.from_prices(
    prices,
    universe,
    cash=cash,
    prior=Prior.time_conditioned(half_life=60),
).with_views(*view_events)

# %%

x = market.viewed_returns()

# %%
for s in x:
    s.plot()
