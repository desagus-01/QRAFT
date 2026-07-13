from collections.abc import Sequence
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.construction.state import PortfolioState
from qraft.core.market import MarketData, MarketDataConfig, Prior
from qraft.core.panel import ScenarioPanel
from qraft.forecast.forecast_paths import AssetUniverse


def asset_universe(*assets: str) -> AssetUniverse:
    return AssetUniverse.factors_free(list(assets))


def price_frame(dates: Sequence[datetime], **prices: Sequence[float]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, **prices})


def cash_frame(dates: Sequence[datetime], rates: Sequence[float]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, "DFF": rates})


def market_data(
    dates: Sequence[datetime],
    *,
    assets: Sequence[str] | None = None,
    cash_rates: Sequence[float] | None = None,
    prior: Prior = Prior.uniform(),
    config: MarketDataConfig = MarketDataConfig(),
    **prices: Sequence[float],
) -> MarketData:
    universe = asset_universe(*(assets or prices.keys()))
    cash = None if cash_rates is None else cash_frame(dates, cash_rates)
    return MarketData.from_prices(
        price_frame(dates, **prices),
        universe,
        cash=cash,
        prior=prior,
        config=config,
    )


def portfolio_state(
    assets: Sequence[str] = ("A",),
    prices: Sequence[float] = (10.0,),
    shares: Sequence[float] = (0.0,),
    cash: float = 100.0,
) -> PortfolioState:
    return PortfolioState(
        asset_order=list(assets),
        initial_prices=np.asarray(prices, dtype=float),
        shares=np.asarray(shares, dtype=float),
        cash=cash,
    )


def scenario_panel_from_levels(
    dates: Sequence[datetime],
    *,
    prob: Sequence[float] | None = None,
    **values: Sequence[float],
) -> ScenarioPanel:
    return ScenarioPanel.from_levels(
        price_frame(dates, **values),
        prob=None if prob is None else np.asarray(prob, dtype=float),
    )


def scenario_panel_from_log_prices(
    dates: Sequence[datetime],
    *,
    prob: Sequence[float] | None = None,
    **values: Sequence[float],
) -> ScenarioPanel:
    return ScenarioPanel.from_log_prices(
        price_frame(dates, **values),
        prob=None if prob is None else np.asarray(prob, dtype=float),
    )


@pytest.fixture
def make_universe():
    return asset_universe


@pytest.fixture
def make_price_frame():
    return price_frame


@pytest.fixture
def make_cash_frame():
    return cash_frame


@pytest.fixture
def make_market_data():
    return market_data


@pytest.fixture
def make_portfolio_state():
    return portfolio_state


@pytest.fixture
def make_scenario_panel_from_levels():
    return scenario_panel_from_levels


@pytest.fixture
def make_scenario_panel_from_log_prices():
    return scenario_panel_from_log_prices
