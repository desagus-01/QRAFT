from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft import Backtest, ForecastSource, Validation
from qraft.backtest.configs import (
    BacktestConfig,
    CombinatorialCVConfig,
    WalkForwardConfig,
)
from qraft.backtest.inputs import PrecomputedInputsProvider
from qraft.core.market import MarketData
from qraft.backtest.selection import __all__ as selection_all
from qraft.backtest.selection.combinatorial import CombinatorialReport
from qraft.backtest.selection.validation import Validation as SelectionValidation
from qraft.backtest.selection.walkforward import WalkForwardReport
from qraft.backtest.simulator import run_backtest
from qraft.construction.optimization.inputs import PolicyInputs
from qraft.construction.policies import EqualWeightPolicy
from qraft.core.schedule import RebalanceSchedule
from qraft.forecast.forecast_paths import AssetUniverse


class _NoopView:
    def apply(self, panel):
        return panel


def _market() -> MarketData:
    return MarketData.from_prices(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, day) for day in range(1, 5)],
                "A": [10.0, 11.0, 12.0, 13.0],
            }
        ),
        AssetUniverse.factors_free(["A"]),
    )


def test_backtest_facade_matches_manual_run_backtest():
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.25)
    config = BacktestConfig(schedule=RebalanceSchedule("every_bar"), initial_cash=100.0)

    facade = Backtest(market=market, policy=policy, config=config).run()
    manual = run_backtest(
        market,
        policy,
        schedule=config.schedule,
        initial_cash=config.initial_cash,
    )

    assert facade.periods == manual.periods
    assert facade.nav_dates == manual.nav_dates
    np.testing.assert_allclose(facade.nav, manual.nav)


def test_backtest_facade_accepts_precomputed_source():
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0)
    config = BacktestConfig(schedule=RebalanceSchedule("every_bar"), initial_cash=100.0)
    source = {
        t: PolicyInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
        )
        for t in market.trading_bars[:-1]
    }

    facade = Backtest(market, policy, source=source, config=config).run()
    manual = run_backtest(
        market,
        policy,
        schedule=config.schedule,
        inputs=PrecomputedInputsProvider(source),
        initial_cash=config.initial_cash,
    )

    np.testing.assert_allclose(facade.nav, manual.nav)


def test_backtest_rejects_current_only_views():
    market = _market().with_current_views(_NoopView())
    policy = EqualWeightPolicy(target_cash_weight=0.25)

    with pytest.raises(ValueError, match="current-only views"):
        Backtest(market=market, policy=policy).run()


def test_validation_dispatches_by_concrete_config(monkeypatch):
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0)
    seen = {}

    def fake_walk_forward(*args, **kwargs):
        seen["walk"] = (args, kwargs)
        return "walk"

    def fake_combinatorial(*args, **kwargs):
        seen["cpcv"] = (args, kwargs)
        return "cpcv"

    monkeypatch.setattr(
        "qraft.backtest.selection.validation.walk_forward", fake_walk_forward
    )
    monkeypatch.setattr(
        "qraft.backtest.selection.validation.combinatorial_purged", fake_combinatorial
    )

    assert (
        SelectionValidation(market, policy, {}, cv_config=WalkForwardConfig()).run()
        == "walk"
    )
    assert (
        SelectionValidation(market, policy, {}, cv_config=CombinatorialCVConfig()).run()
        == "cpcv"
    )

    assert isinstance(seen["walk"][1]["walk_config"], WalkForwardConfig)
    assert isinstance(seen["cpcv"][1]["cv_config"], CombinatorialCVConfig)


def test_validation_rejects_current_only_views():
    market = _market().with_current_views(_NoopView())
    policy = EqualWeightPolicy(target_cash_weight=0.0)

    with pytest.raises(ValueError, match="current-only views"):
        SelectionValidation(market, policy, {}, cv_config=WalkForwardConfig()).run()


def test_public_facade_exports():
    assert Backtest.__name__ == "Backtest"
    assert Validation.__name__ == "Validation"
    assert ForecastSource is not None
    assert WalkForwardReport is not None
    assert CombinatorialReport is not None
    assert "combinatorial_purged" in selection_all
    assert "run_combinatorial_purged" not in selection_all
