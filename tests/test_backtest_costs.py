from datetime import datetime

import numpy as np
import polars as pl

from qraft.backtest.costs import CostModel
from qraft.backtest.market import MarketData
from qraft.backtest.schedule import RebalanceSchedule
from qraft.backtest.simulator import run_backtest
from qraft.construction.optimization.objectives.specs import (
    HoldingCost,
    TransactionCost,
    transaction_cost_coeffs,
    transaction_cost_value,
)
from qraft.construction.policies import EqualWeightPolicy
from qraft.forecast.forecast_paths import AssetUniverse


DATES = [datetime(2024, 1, d) for d in (1, 2, 3, 4)]


def _market() -> MarketData:
    return MarketData.from_prices(
        pl.DataFrame(
            {
                "date": DATES,
                "A": [10.0, 12.0, 12.0, 15.0],
                "B": [20.0, 18.0, 18.0, 18.0],
            }
        ),
        AssetUniverse.factors_free(["A", "B"]),
        cash=pl.DataFrame({"date": DATES, "DFF": [3.6, 7.2, 7.2, 7.2]}),
    )


def _run(transaction=None, holding=None):
    return run_backtest(
        _market(),
        EqualWeightPolicy(target_cash_weight=0.25),
        schedule=RebalanceSchedule(cadence="every_bar"),
        initial_cash=100.0,
        costs=CostModel(transaction=transaction, holding=holding),
    )


def test_frictionless_default_for_costless_policy():
    a = run_backtest(
        _market(),
        EqualWeightPolicy(0.25),
        schedule=RebalanceSchedule(cadence="every_bar"),
    )
    b = _run(TransactionCost(cost=0.0))
    np.testing.assert_allclose(a.nav, b.nav, rtol=1e-12)


def test_linear_cost_matches_traded_notional():
    rate = 0.001
    for p in _run(TransactionCost(cost=rate, market_impact=0.0)).periods:
        notional = float(
            np.abs(p.executed_share_trades * p.state_before.initial_prices).sum()
        )
        np.testing.assert_allclose(p.cost, rate * notional, rtol=1e-10)


def test_self_financing_with_cost():
    for p in _run(
        TransactionCost(cost=0.001, pershare_cost=0.01, market_impact=0.0)
    ).periods:
        trade_value = float(p.executed_share_trades @ p.state_before.initial_prices)
        np.testing.assert_allclose(
            p.state_after.cash - p.state_before.cash,
            -trade_value - p.cost,
            rtol=1e-10,
        )


def test_all_cash_pays_no_holding_cost():
    from qraft.backtest.baselines import AllCashPolicy

    res = run_backtest(
        _market(),
        AllCashPolicy(),
        schedule=RebalanceSchedule(cadence="every_bar"),
        costs=CostModel(holding=HoldingCost(0.02, 0.02, 0.0)),
    )
    np.testing.assert_allclose(res.holding_costs, 0.0, atol=1e-12)


def test_long_book_accrues_holding_drag():
    res = run_backtest(
        _market(),
        EqualWeightPolicy(target_cash_weight=0.0),
        schedule=RebalanceSchedule(cadence="every_bar"),
        costs=CostModel(holding=HoldingCost(0.0, 0.0252, 0.0)),
    )
    assert res.holding_costs[0] == 0.0
    assert res.total_holding_cost > 0.0
    assert len(res.holding_costs) == len(res.nav)


def test_cost_model_from_policy_reads_preset():
    from qraft.construction.policies import MPOPolicy

    pol = MPOPolicy.preset("mean_covariance", risk_aversion=1.0)
    model = CostModel.from_policy(pol)
    assert isinstance(model.transaction, TransactionCost)
    assert isinstance(model.holding, HoldingCost)


def test_frictionless_override():
    res = run_backtest(
        _market(),
        EqualWeightPolicy(0.25),
        schedule=RebalanceSchedule(cadence="every_bar"),
        costs=CostModel.frictionless(),
    )
    np.testing.assert_allclose(res.period_costs, 0.0, atol=1e-12)
    np.testing.assert_allclose(res.holding_costs, 0.0, atol=1e-12)


def test_none_equals_frictionless():
    free = _run(None)
    zero = _run(TransactionCost(cost=0.0))
    np.testing.assert_allclose(free.nav, zero.nav, rtol=1e-12)
    np.testing.assert_allclose(zero.period_costs, 0.0, atol=1e-12)


def test_per_share_cost():
    for p in _run(
        TransactionCost(cost=0.0, pershare_cost=0.05, market_impact=0.0)
    ).periods:
        shares = float(np.abs(p.executed_share_trades).sum())
        np.testing.assert_allclose(p.cost, 0.05 * shares, rtol=1e-10)


def test_cost_is_nav_drag():
    free = _run(None)
    costed = _run(TransactionCost(cost=0.001, market_impact=0.0))
    assert np.all(costed.nav <= free.nav + 1e-12)


def test_shared_definition_parity():
    spec = TransactionCost(cost=0.0005, market_impact=1.0, exponent=1.5)
    z = np.array([0.10, -0.05])
    sigma = np.array([0.01, 0.02])
    linear, impact, bias = transaction_cost_coeffs(spec, n_assets=2, sigma=sigma)
    manual = 100.0 * (linear @ np.abs(z) + impact @ (np.abs(z) ** 1.5) + bias @ z)
    got = transaction_cost_value(spec, z, nav=100.0, sigma=sigma)
    np.testing.assert_allclose(got, manual, rtol=1e-12)


def test_impact_uses_forecast_sigma():
    spec = TransactionCost(cost=0.0, market_impact=1.0, exponent=1.5)
    z = np.array([0.2])
    np.testing.assert_allclose(
        transaction_cost_value(spec, z, nav=1_000.0, sigma=np.array([0.015])),
        1_000.0 * 1.0 * 0.015 * (0.2**1.5),
        rtol=1e-12,
    )
