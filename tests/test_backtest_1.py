from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
import pytest

from qraft.backtest.baselines import AllCashPolicy, NoTradePolicy
from qraft.backtest.execution import BacktestResult
from qraft.core.market import HistoryWeighting, MarketData
from qraft.core.snapshot import MarketSnapshot
from qraft.backtest.simulator import run_backtest
from qraft.construction.optimization.inputs import PolicyInputs
from qraft.construction.optimization.optimization import MPOFailure, OptimizationFailure
from qraft.construction.policies import EqualWeightPolicy, PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.core.schedule import RebalanceSchedule
from qraft.forecast.forecast_paths import AssetUniverse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _universe(*assets: str) -> AssetUniverse:
    return AssetUniverse.factors_free(list(assets))


def _price_frame(dates, **prices) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, **prices})


def _cash_frame(dates, rates: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, "DFF": rates})


DATES_4 = [
    datetime(2024, 1, 1),
    datetime(2024, 1, 2),
    datetime(2024, 1, 3),
    datetime(2024, 1, 4),
]

DATES_6 = [
    datetime(2024, 1, 1),
    datetime(2024, 1, 2),
    datetime(2024, 1, 3),
    datetime(2024, 1, 4),
    datetime(2024, 1, 5),
    datetime(2024, 1, 8),
]


class RecordingForecaster:
    """Records received snapshots; returns flat PolicyInputs (a provider)."""

    def __init__(self) -> None:
        self.snapshots: list[MarketSnapshot] = []

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
        self.snapshots.append(snapshot)
        n = len(snapshot.universe.assets)
        return PolicyInputs.from_arrays(
            assets=snapshot.universe.assets,
            mean=np.ones((1, n)),
            cash_return=np.array([0.0]),
        )


class FailingPolicy:
    name: str = "failing"
    min_history: int = 0

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: PolicyInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        raise RuntimeError("intentional failure for testing")


class OptimizationFailingPolicy:
    name: str = "optimization_failing"
    min_history: int = 0

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: PolicyInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        raise OptimizationFailure(
            MPOFailure(
                status="cvar_cutting_plane_nonconverged",
                message="Optimization failed: cvar_cutting_plane_nonconverged",
            )
        )


# ---------------------------------------------------------------------------
# 1. Basic lagged execution & period recording
# ---------------------------------------------------------------------------


def test_1_lagged_execution_and_periods() -> None:
    forecaster = RecordingForecaster()
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=forecaster,
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    assert result.policy_name == "equal_weight"
    assert result.asset_order == ["A", "B"]
    assert result.nav_dates == market.trading_bars
    np.testing.assert_allclose(
        result.nav,
        [100.0, 100.01, 100.0150005, 109.39640821875],
    )
    assert len(result.periods) == 3
    first = result.periods[0]
    assert first.decision_bar == datetime(2024, 1, 1)
    assert first.execution_bar == datetime(2024, 1, 2)
    np.testing.assert_allclose(first.state_before.initial_prices, [12.0, 18.0])
    np.testing.assert_allclose(first.executed_share_trades, [3.1253125, 2.083541667])
    np.testing.assert_allclose(first.state_after.asset_weights, [0.375, 0.375])
    assert float(first.state_after.cash_weight) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 2. Only bars with a next bar are used as decision bars
# ---------------------------------------------------------------------------


def test_2_only_executable_decision_bars() -> None:
    forecaster = RecordingForecaster()
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=forecaster,
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    assert [s.t for s in forecaster.snapshots] == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]
    assert [s.t_next for s in forecaster.snapshots] == [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
        datetime(2024, 1, 4),
    ]
    assert [s.history.dates.max() for s in forecaster.snapshots] == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]


# ---------------------------------------------------------------------------
# 3. AllCashPolicy — NAV changes only from cash compounding
# ---------------------------------------------------------------------------


def test_3_all_cash_policy() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [5.0, 5.0, 5.0, 5.0]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=AllCashPolicy(),
        initial_cash=100.0,
    )

    assert result.policy_name == "all_cash"
    assert len(result.periods) == 3
    # All cash — no share trades ever
    for p in result.periods:
        np.testing.assert_allclose(p.executed_share_trades, [0.0, 0.0], atol=1e-12)
        assert float(p.state_after.cash_weight) == pytest.approx(1.0)

    # NAV grows only from cash compounding: 5% annual DFF
    # day 1→2: 5% * 1/360 ≈ 0.0001389% per day → 100 * (1 + 0.05/360) = 100.0138889
    # day 2→3: same rate → 100.0138889 * (1 + 0.05/360)
    # day 3→4: same rate
    expected = 100.0
    for _ in range(3):
        expected *= 1.0 + 0.05 / 360
    np.testing.assert_allclose(result.nav[-1], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. NoTradePolicy — portfolio drifts with price moves only
# ---------------------------------------------------------------------------


def test_5_no_trade_policy() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=NoTradePolicy(),
        initial_cash=100.0,
    )

    assert result.policy_name == "no_trade"
    # NoTradePolicy targets current weights → zero turnover
    for p in result.periods:
        np.testing.assert_allclose(p.executed_share_trades, [0.0, 0.0], atol=1e-12)
    # NAV starts at 100 and changes only from price moves and cash compounding
    assert result.nav[0] == 100.0
    assert result.nav[-1] > 100.0  # positive price drift + cash


# ---------------------------------------------------------------------------
# 6. No cash data — cash_rate defaults to 0.0
# ---------------------------------------------------------------------------


def test_6_no_cash_data() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=None,
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    # Fully invested, no cash compounding
    assert result.nav[0] == 100.0
    # After rebalance to equal weight, NAV tracks asset prices
    assert len(result.nav) == 4


# ---------------------------------------------------------------------------
# 7. Solver resilience — policy exception triggers hold fallback
# ---------------------------------------------------------------------------


def test_7_solver_fallback_on_policy_failure() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=FailingPolicy(),
        initial_cash=100.0,
    )

    # Should complete without crashing
    assert len(result.nav) == 4
    # All periods should have solver_status indicating error
    for p in result.periods:
        assert p.solver_status == "solver_error"
        np.testing.assert_allclose(p.executed_share_trades, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.period_turnovers, 0.0, atol=1e-12)


def test_optimizer_failure_status_is_preserved_when_holding() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=OptimizationFailingPolicy(),
        initial_cash=100.0,
    )

    assert {p.solver_status for p in result.periods} == {
        "cvar_cutting_plane_nonconverged"
    }
    for p in result.periods:
        np.testing.assert_allclose(p.executed_share_trades, [0.0, 0.0], atol=1e-12)


def test_period_turnover_includes_cash_leg() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 10.0, 10.0, 10.0], B=[20.0, 20.0, 20.0, 20.0]),
        _universe("A", "B"),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    first = result.periods[0]
    risky_trade = float(
        np.abs(first.executed_share_trades * first.state_before.initial_prices).sum()
    )
    cash_trade = float(first.state_after.cash - first.state_before.cash)
    expected = (
        0.5 * (risky_trade + abs(cash_trade)) / first.state_before.portfolio_value
    )

    assert result.period_turnovers[0] == pytest.approx(expected)
    assert result.period_turnovers[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. Month-end cadence
# ---------------------------------------------------------------------------


def test_7_month_end_cadence() -> None:
    dates = [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
        datetime(2024, 1, 31),
        datetime(2024, 2, 1),
        datetime(2024, 2, 29),
        datetime(2024, 3, 1),
    ]
    n = len(dates)
    market = MarketData.from_prices(
        _price_frame(dates, A=list(range(100, 100 + n)), B=list(range(200, 200 + n))),
        _universe("A", "B"),
        cash=_cash_frame(dates, [5.0] * n),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="month_end"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    # Decision bars: Jan 31, Feb 29 → 2 periods (last bar Mar 1 has no next bar)
    assert len(result.periods) == 2
    assert result.periods[0].decision_bar == datetime(2024, 1, 31)
    assert result.periods[1].decision_bar == datetime(2024, 2, 29)


# ---------------------------------------------------------------------------
# 8. Quarter-end cadence
# ---------------------------------------------------------------------------


def test_8_quarter_end_cadence() -> None:
    dates = [
        datetime(2024, 1, 2),
        datetime(2024, 3, 29),
        datetime(2024, 4, 1),
        datetime(2024, 6, 28),
        datetime(2024, 7, 1),
    ]
    n = len(dates)
    market = MarketData.from_prices(
        _price_frame(dates, A=list(range(100, 100 + n)), B=list(range(200, 200 + n))),
        _universe("A", "B"),
        cash=_cash_frame(dates, [5.0] * n),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="quarter_end"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    # Decision bars: Mar 29, Jun 28 → 2 periods
    assert len(result.periods) == 2
    assert result.periods[0].decision_bar == datetime(2024, 3, 29)
    assert result.periods[1].decision_bar == datetime(2024, 6, 28)


# ---------------------------------------------------------------------------
# 9. Self-financing property — trades are cash-neutral
# ---------------------------------------------------------------------------


def test_9_self_financing_trades() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    for p in result.periods:
        # trade_value = Σ executed_shares * execution_price
        prices = p.state_before.initial_prices
        trade_value = float(p.executed_share_trades @ prices)
        # cash delta should offset trade_value (self-financing)
        cash_delta = p.state_after.cash - p.state_before.cash
        np.testing.assert_allclose(cash_delta, -trade_value, rtol=1e-10)


# ---------------------------------------------------------------------------
# 10. Single asset
# ---------------------------------------------------------------------------


def test_10_single_asset() -> None:
    dates = DATES_4
    market = MarketData.from_prices(
        _price_frame(dates, A=[15.0, 16.5, 18.0, 19.5]),
        _universe("A"),
        cash=_cash_frame(dates, [5.0] * 4),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    assert result.asset_order == ["A"]
    assert len(result.periods) == 3
    # Fully invested in one asset
    for p in result.periods:
        np.testing.assert_allclose(p.state_after.asset_weights, [1.0])
        assert float(p.state_after.cash_weight) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. Three assets with factors (factors excluded from trading)
# ---------------------------------------------------------------------------


def test_11_three_assets() -> None:
    dates = DATES_4
    market = MarketData.from_prices(
        _price_frame(
            dates,
            A=[10.0, 11.0, 12.0, 13.0],
            B=[20.0, 19.0, 18.0, 17.0],
            C=[30.0, 31.0, 32.0, 33.0],
        ),
        _universe("A", "B", "C"),
        cash=_cash_frame(dates, [5.0] * 4),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    assert result.asset_order == ["A", "B", "C"]
    assert len(result.nav) == 4
    for p in result.periods:
        w = p.state_after.asset_weights
        np.testing.assert_allclose(w, [0.25, 0.25, 0.25], atol=1e-10)
        assert float(p.state_after.cash_weight) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 12. MarketData constructed from raw prices
# ---------------------------------------------------------------------------


def test_12_from_prices() -> None:
    dates = DATES_4
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": [10.0, 12.0, 12.0, 15.0],
            "B": [20.0, 18.0, 18.0, 18.0],
        }
    )
    market = MarketData.from_prices(
        prices, _universe("A", "B"), cash=_cash_frame(dates, [3.6, 7.2, 7.2, 7.2])
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    np.testing.assert_allclose(
        result.nav,
        [100.0, 100.01, 100.0150005, 109.39640821875],
    )
    assert len(result.periods) == 3


# ---------------------------------------------------------------------------
# 13. NAV correctness manual verification
# ---------------------------------------------------------------------------


def test_13_nav_correctness() -> None:
    dates = DATES_4
    prices_a = [10.0, 12.0, 12.0, 15.0]
    prices_b = [20.0, 18.0, 18.0, 18.0]
    market = MarketData.from_prices(
        _price_frame(dates, A=prices_a, B=prices_b),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [5.0, 5.0, 5.0, 5.0]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    # Fully invested equal-weight 2 assets: 50/50 at every rebalance
    # DFF=5%, compounding daily ACT/360: daily factor = 1 + 0.05/360 ≈ 1.000138889
    # bar 0 (2024-01-01): NAV=100, shares=[0,0], cash=100
    # bar 1 (2024-01-02): cash compounds → 100 * 1.000138889 = 100.0138889
    #   rebalance from all-cash to 50/50 equal weight
    #   nav_before_exec = 100.0138889, target each = 50.00694445
    #   shares_A = 50.00694445 / 12 = 4.16724537
    #   shares_B = 50.00694445 / 18 = 2.77816358
    #   cash = 0 (self-financing)
    #   NAV = 4.16724537*12 + 2.77816358*18 = 100.0138889
    # bar 2 (2024-01-03): prices unchanged [12,18], cash compounds (still 0)
    #   already 50/50 → no trade, NAV unchanged = 100.0138889
    # bar 3 (2024-01-04): prices [15, 18], cash compounds (still 0)
    #   NAV = 4.16724537*15 + 2.77816358*18 = 62.5086806 + 50.0069444 = 112.515625

    expected_nav = [100.0, 100.01388888888889, 100.01388888888889, 112.515625]
    np.testing.assert_allclose(result.nav, expected_nav, rtol=1e-10)
    assert len(result.periods) == 3


# ---------------------------------------------------------------------------
# 14. Week-end cadence
# ---------------------------------------------------------------------------


def test_14_week_end_cadence() -> None:
    dates = [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
        datetime(2024, 1, 5),  # Fri → week-end
        datetime(2024, 1, 8),
        datetime(2024, 1, 12),  # Fri → week-end
        datetime(2024, 1, 15),
    ]
    n = len(dates)
    market = MarketData.from_prices(
        _price_frame(dates, A=list(range(100, 100 + n)), B=list(range(200, 200 + n))),
        _universe("A", "B"),
        cash=_cash_frame(dates, [5.0] * n),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="week_end"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    assert len(result.periods) == 2
    assert result.periods[0].decision_bar == datetime(2024, 1, 5)
    assert result.periods[1].decision_bar == datetime(2024, 1, 12)


# ---------------------------------------------------------------------------
# 15. BacktestResult structure
# ---------------------------------------------------------------------------


def test_15_backtest_result_structure() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    assert isinstance(result, BacktestResult)
    assert isinstance(result.nav, np.ndarray)
    assert len(result.nav_dates) == len(result.nav)
    assert len(result.periods) <= len(result.nav_dates)


# ---------------------------------------------------------------------------
# 16. Forecasting integration — forecaster receives correct universe
# ---------------------------------------------------------------------------


def test_16_forecaster_receives_correct_universe() -> None:
    received_universes: list[AssetUniverse] = []

    class CapturingForecaster:
        def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
            received_universes.append(snapshot.universe)
            n = len(snapshot.universe.assets)
            return PolicyInputs.from_arrays(
                assets=snapshot.universe.assets,
                mean=np.ones((1, n)),
                cash_return=np.array([0.0]),
            )

    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=CapturingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    assert len(received_universes) == 3
    for u in received_universes:
        assert u.assets == ["A", "B"]
        assert u.factors == []


# ---------------------------------------------------------------------------
# 17. HistoryWeighting with state_smooth does not break backtest
# ---------------------------------------------------------------------------


def test_17_state_smooth_weighting() -> None:
    dates = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
        datetime(2024, 1, 4),
        datetime(2024, 1, 5),
        datetime(2024, 1, 8),
    ]
    n = len(dates)
    market = MarketData.from_prices(
        _price_frame(dates, A=list(range(100, 100 + n)), B=list(range(200, 200 + n))),
        _universe("A", "B"),
        cash=_cash_frame(dates, [5.0] * n),
        history_weighting=HistoryWeighting(scheme="state_smooth", half_life=63),
    )

    # History must have enough points for weighting to work
    market.history_through(dates[2])

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        inputs=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    assert len(result.nav) == n
