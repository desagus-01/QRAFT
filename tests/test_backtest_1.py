from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.backtest.baselines import AllCashPolicy, NoTradePolicy
from qraft.backtest.execution import BacktestResult
from qraft.backtest.market import MarketData, MarketSnapshot, WindowWeighting
from qraft.backtest.schedule import RebalanceSchedule
from qraft.backtest.simulator import run_backtest
from qraft.construction.policies import EqualWeightPolicy, PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths

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
    """Records received snapshots; returns flat two-step forecasts."""

    def __init__(self) -> None:
        self.snapshots: list[MarketSnapshot] = []

    def forecast(
        self, snapshot: MarketSnapshot, universe: AssetUniverse
    ) -> ForecastPaths:
        self.snapshots.append(snapshot)
        return ForecastPaths(
            asset_paths={
                asset: np.array([[snapshot.prices_t[i]], [snapshot.prices_t[i]]])
                for i, asset in enumerate(universe.assets)
            },
            dates=pl.Series("date", [snapshot.t_next]),
            path_probs=np.array([0.5, 0.5]),
            initial_prices=dict(zip(universe.assets, snapshot.prices_t)),
            universe=universe,
        )


class FailingPolicy:
    name: str = "failing"

    def decide(self, state: PortfolioState, forecasts: ForecastPaths) -> PolicyDecision:
        raise RuntimeError("intentional failure for testing")


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
        forecaster=forecaster,
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
    assert first.forecasts is not None


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
        forecaster=forecaster,
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
# 3. Forecast storage can be skipped
# ---------------------------------------------------------------------------


def test_3_can_skip_storing_forecasts() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
        store_forecasts=False,
    )

    assert result.periods
    assert all(p.forecasts is None for p in result.periods)


# ---------------------------------------------------------------------------
# 4. AllCashPolicy — NAV changes only from cash compounding
# ---------------------------------------------------------------------------


def test_4_all_cash_policy() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [5.0, 5.0, 5.0, 5.0]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=RecordingForecaster(),
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
        forecaster=RecordingForecaster(),
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
        forecaster=RecordingForecaster(),
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
        forecaster=RecordingForecaster(),
        policy=FailingPolicy(),
        initial_cash=100.0,
    )

    # Should complete without crashing
    assert len(result.nav) == 4
    # All periods should have solver_status indicating error
    for p in result.periods:
        assert p.solver_status == "solver_error"


# ---------------------------------------------------------------------------
# 8. Month-end cadence
# ---------------------------------------------------------------------------


def test_8_month_end_cadence() -> None:
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
        forecaster=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    # Decision bars: Jan 31, Feb 29 → 2 periods (last bar Mar 1 has no next bar)
    assert len(result.periods) == 2
    assert result.periods[0].decision_bar == datetime(2024, 1, 31)
    assert result.periods[1].decision_bar == datetime(2024, 2, 29)


# ---------------------------------------------------------------------------
# 9. Quarter-end cadence
# ---------------------------------------------------------------------------


def test_9_quarter_end_cadence() -> None:
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
        forecaster=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    # Decision bars: Mar 29, Jun 28 → 2 periods
    assert len(result.periods) == 2
    assert result.periods[0].decision_bar == datetime(2024, 3, 29)
    assert result.periods[1].decision_bar == datetime(2024, 6, 28)


# ---------------------------------------------------------------------------
# 10. Self-financing property — trades are cash-neutral
# ---------------------------------------------------------------------------


def test_10_self_financing_trades() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=RecordingForecaster(),
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
# 11. Single asset
# ---------------------------------------------------------------------------


def test_11_single_asset() -> None:
    dates = DATES_4
    market = MarketData.from_prices(
        _price_frame(dates, A=[10.0, 11.0, 12.0, 13.0]),
        _universe("A"),
        cash=_cash_frame(dates, [5.0] * 4),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=RecordingForecaster(),
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
# 12. Three assets with factors (factors excluded from trading)
# ---------------------------------------------------------------------------


def test_12_three_assets() -> None:
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
        forecaster=RecordingForecaster(),
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
# 13. MarketData constructed from log prices
# ---------------------------------------------------------------------------


def test_13_from_log_prices() -> None:
    dates = DATES_4
    log_prices = pl.DataFrame(
        {
            "date": dates,
            "A": [np.log(10.0), np.log(12.0), np.log(12.0), np.log(15.0)],
            "B": [np.log(20.0), np.log(18.0), np.log(18.0), np.log(18.0)],
        }
    )
    market = MarketData.from_log_prices(
        log_prices, _universe("A", "B"), cash=_cash_frame(dates, [3.6, 7.2, 7.2, 7.2])
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    # Should match from_prices with the same exp values
    np.testing.assert_allclose(
        result.nav,
        [100.0, 100.01, 100.0150005, 109.39640821875],
    )
    assert len(result.periods) == 3


# ---------------------------------------------------------------------------
# 14. NAV correctness manual verification
# ---------------------------------------------------------------------------


def test_14_nav_correctness() -> None:
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
        forecaster=RecordingForecaster(),
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
# 15. Week-end cadence
# ---------------------------------------------------------------------------


def test_15_week_end_cadence() -> None:
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
        forecaster=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    assert len(result.periods) == 2
    assert result.periods[0].decision_bar == datetime(2024, 1, 5)
    assert result.periods[1].decision_bar == datetime(2024, 1, 12)


# ---------------------------------------------------------------------------
# 16. BacktestResult structure
# ---------------------------------------------------------------------------


def test_16_backtest_result_structure() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    assert isinstance(result, BacktestResult)
    assert isinstance(result.nav, np.ndarray)
    assert len(result.nav_dates) == len(result.nav)
    assert len(result.periods) <= len(result.nav_dates)


# ---------------------------------------------------------------------------
# 17. Forecasting integration — forecaster receives correct universe
# ---------------------------------------------------------------------------


def test_17_forecaster_receives_correct_universe() -> None:
    received_universes: list[AssetUniverse] = []

    class CapturingForecaster:
        def forecast(
            self, snapshot: MarketSnapshot, universe: AssetUniverse
        ) -> ForecastPaths:
            received_universes.append(universe)
            return ForecastPaths(
                asset_paths={a: np.array([[1.0], [1.0]]) for a in universe.assets},
                dates=pl.Series("date", [snapshot.t_next]),
                path_probs=np.array([0.5, 0.5]),
                initial_prices=dict(zip(universe.assets, snapshot.prices_t)),
                universe=universe,
            )

    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=CapturingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    assert len(received_universes) == 3
    for u in received_universes:
        assert u.assets == ["A", "B"]
        assert u.factors == []


# ---------------------------------------------------------------------------
# 18. WindowWeighting with state_smooth does not break backtest
# ---------------------------------------------------------------------------


def test_18_state_smooth_weighting() -> None:
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
        weighting=WindowWeighting(scheme="state_smooth", half_life=63),
    )

    # History must have enough points for weighting to work
    market.history_through(dates[2])

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.0),
        initial_cash=100.0,
    )

    assert len(result.nav) == n


# ---------------------------------------------------------------------------
# 19. Default step_size stores forecasts (regression)
# ---------------------------------------------------------------------------


def test_19_forecasts_stored_by_default() -> None:
    market = MarketData.from_prices(
        _price_frame(DATES_4, A=[10.0, 12.0, 12.0, 15.0], B=[20.0, 18.0, 18.0, 18.0]),
        _universe("A", "B"),
        cash=_cash_frame(DATES_4, [3.6, 7.2, 7.2, 7.2]),
    )

    result = run_backtest(
        market=market,
        schedule=RebalanceSchedule(cadence="every_bar"),
        forecaster=RecordingForecaster(),
        policy=EqualWeightPolicy(target_cash_weight=0.25),
        initial_cash=100.0,
    )

    assert result.periods
    assert all(p.forecasts is not None for p in result.periods)
