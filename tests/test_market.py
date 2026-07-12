from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.core.market import (
    MarketData,
    MarketDataConfig,
    Prior,
)
from qraft.core.panel import ScenarioPanel
from qraft.core.scenarios.transforms import Views
from qraft.core.scenarios.view_types import MeanView
from qraft.core.scenarios.views import ViewEvent
from qraft.core.snapshot import MarketSnapshot
from qraft.forecast.forecast_paths import AssetUniverse


def _universe() -> AssetUniverse:
    return AssetUniverse.factors_free(["A", "B"])


def _price_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "A": [10.0, 11.0, 12.0],
            "B": [20.0, 22.0, 24.0],
        }
    )


def _cash_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "DFF": [5.0, 5.25, 5.5],
        }
    )


def _market_data() -> MarketData:
    return MarketData.from_prices(_price_frame(), _universe(), cash=_cash_frame())


class _ScaleProbView:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def apply(self, panel: ScenarioPanel) -> ScenarioPanel:
        weights = np.arange(1, panel.values.height + 1, dtype=float) ** self.factor
        return panel.with_prob(weights / weights.sum())


class _CapturePanelView:
    def __init__(self) -> None:
        self.panel: ScenarioPanel | None = None

    def apply(self, panel: ScenarioPanel) -> ScenarioPanel:
        self.panel = panel
        return panel


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


def test_market_snapshot_dataclass() -> None:
    panel = ScenarioPanel.from_prices(_price_frame())
    snap = MarketSnapshot(
        t=datetime(2024, 1, 2),
        t_next=datetime(2024, 1, 3),
        universe=_universe(),
        history=panel,
        prices_t=np.array([11.0, 22.0]),
        cash_rate=0.001,
    )
    assert snap.t == datetime(2024, 1, 2)
    assert snap.t_next == datetime(2024, 1, 3)
    assert snap.universe.assets == ["A", "B"]
    assert snap.history is panel
    np.testing.assert_allclose(snap.prices_t, [11.0, 22.0])
    assert snap.cash_rate == 0.001


def test_market_data_config_defaults() -> None:
    cfg = MarketDataConfig()
    assert cfg.cash_column == "DFF"
    assert cfg.periods_per_year is None
    assert cfg.cash_day_count == 360


def test_market_data_rejects_non_positive_prices() -> None:
    df = _price_frame().with_columns(pl.lit(0.0).alias("A"))

    with pytest.raises(ValueError, match="prices must be strictly positive"):
        MarketData.from_prices(df, _universe())


def test_market_data_rejects_non_finite_cash_rates() -> None:
    cash = _cash_frame().with_columns(pl.lit(float("nan")).alias("DFF"))

    with pytest.raises(ValueError, match="Cash rates"):
        MarketData.from_prices(_price_frame(), _universe(), cash=cash)


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


def test_prior_default_scheme_is_uniform() -> None:
    w = Prior.uniform()
    assert w.scheme == "uniform"
    assert w.half_life is None


def test_prior_state_smooth_requires_half_life() -> None:
    with pytest.raises(ValueError, match="half_life"):
        Prior(scheme="state_smooth")


def test_prior_time_conditioned_accepted_with_half_life() -> None:
    w = Prior.time_conditioned(half_life=60)
    assert w.scheme == "state_smooth"
    assert w.half_life == 60


def test_prior_uniform_probs() -> None:
    probs = Prior.uniform().probs(5)
    assert len(probs) == 5
    np.testing.assert_allclose(probs, np.full(5, 0.2))


def test_prior_time_conditioned_probs_sums_to_one() -> None:
    probs = Prior.time_conditioned(half_life=60).probs(10)
    assert len(probs) == 10
    np.testing.assert_allclose(probs.sum(), 1.0)


# ---------------------------------------------------------------------------
# MarketData – construction
# ---------------------------------------------------------------------------


def test_from_prices_validates_date_column() -> None:
    with pytest.raises(ValueError, match="date"):
        MarketData.from_prices(pl.DataFrame({"A": [1.0]}), _universe())


def test_from_prices_sorts_by_date() -> None:
    universe = AssetUniverse.factors_free(["A"])
    df = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 3),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            ],
            "A": [17.0, 15.0, 16.0],
        }
    )
    md = MarketData.from_prices(df, universe)
    assert md.trading_bars == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]


def test_from_prices_with_cash() -> None:
    md = _market_data()
    assert md.cash is not None
    assert md.cash.columns == ["date", "DFF"]


def test_from_prices_without_cash() -> None:
    md = MarketData.from_prices(_price_frame(), _universe())
    assert md.cash is None


def test_from_prices_rejects_log_price_like_values() -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "A": [np.log(10.0), np.log(11.0)],
            "B": [np.log(20.0), np.log(22.0)],
        }
    )
    with pytest.raises(ValueError, match="values look like log prices"):
        MarketData.from_prices(df, _universe())


# ---------------------------------------------------------------------------
# MarketData – trading_bars & prices_at
# ---------------------------------------------------------------------------


def test_trading_bars() -> None:
    md = _market_data()
    assert md.trading_bars == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]


def test_prices_at() -> None:
    md = _market_data()
    np.testing.assert_allclose(md.prices_at(datetime(2024, 1, 2)), [11.0, 22.0])


def test_prices_at_accepts_string_date() -> None:
    md = _market_data()
    np.testing.assert_allclose(md.prices_at("2024-01-02"), [11.0, 22.0])


def test_prices_at_raises_on_missing_date() -> None:
    md = _market_data()
    with pytest.raises(ValueError, match="No market data"):
        md.prices_at(datetime(2024, 1, 5))


# ---------------------------------------------------------------------------
# MarketData – history_through
# ---------------------------------------------------------------------------


def test_history_through_returns_scenario_panel() -> None:
    md = _market_data()
    panel = md.history_through(datetime(2024, 1, 2))
    assert isinstance(panel, ScenarioPanel)
    assert panel.kind == "log_price"


def test_history_through_accepts_string_date() -> None:
    md = _market_data()
    panel = md.history_through("2024-01-02")
    assert panel.dates.to_list() == [datetime(2024, 1, 1), datetime(2024, 1, 2)]


def test_history_through_raises_on_no_data() -> None:
    md = _market_data()
    with pytest.raises(ValueError, match="No history"):
        md.history_through(datetime(2020, 1, 1))


def test_with_views_applies_latest_causal_event_only() -> None:
    md = _market_data().with_views(
        (datetime(2024, 1, 2), _ScaleProbView(1.0)),
        ViewEvent(datetime(2024, 1, 3), _ScaleProbView(2.0)),
    )

    before = md.history_through(datetime(2024, 1, 1))
    first = md.history_through(datetime(2024, 1, 2))
    second = md.history_through(datetime(2024, 1, 3))

    np.testing.assert_allclose(before.prob, [1.0])
    np.testing.assert_allclose(first.prob, [0.5, 0.5])
    np.testing.assert_allclose(second.prob, [1 / 3, 2 / 15, 8 / 15])


def test_view_events_receive_simple_return_history() -> None:
    view = _CapturePanelView()
    md = _market_data().with_views((datetime(2024, 1, 3), view))

    panel = md.history_through(datetime(2024, 1, 3))

    assert view.panel is not None
    assert panel.kind == "log_price"
    assert view.panel.kind == "return"
    assert panel.dates.to_list() == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]
    assert view.panel.dates.to_list() == [datetime(2024, 1, 2), datetime(2024, 1, 3)]
    np.testing.assert_allclose(
        view.panel.values.select("A", "B").to_numpy(),
        [[0.1, 0.1], [1.0 / 11.0, 1.0 / 11.0]],
    )


def test_history_through_expands_view_posterior_to_log_price_history() -> None:
    md = _market_data().with_views(
        (datetime(2024, 1, 3), Views([MeanView("A", "==", 1.0 / 11.0)]))
    )

    panel = md.history_through(datetime(2024, 1, 3))
    viewed = md.viewed_returns(t=datetime(2024, 1, 3))

    assert panel.kind == "log_price"
    assert panel.dates.to_list() == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]
    np.testing.assert_allclose(panel.prob, viewed.prob_for(panel))
    assert len(panel.prob) == 3
    assert len(viewed.posterior) == 2


def test_viewed_returns_exposes_simple_return_distribution() -> None:
    md = _market_data()
    views = Views([MeanView("A", "==", 0.1)])

    viewed = md.viewed_returns(views, datetime(2024, 1, 3))
    via_view = views.against(md)

    assert viewed.panel.kind == "return"
    assert viewed.panel.dates.to_list() == [datetime(2024, 1, 2), datetime(2024, 1, 3)]
    np.testing.assert_allclose(
        viewed.panel.values.get_column("A").to_numpy(), [0.1, 1.0 / 11.0]
    )
    assert viewed.as_of == datetime(2024, 1, 3)
    assert set(viewed.moments()["A"]) == {
        "prior_mean",
        "posterior_mean",
        "prior_std",
        "posterior_std",
    }
    np.testing.assert_allclose(viewed.posterior, via_view.posterior)


def test_viewed_returns_without_args_returns_each_registered_view_date() -> None:
    first = Views([MeanView("A", "==", 0.1)])
    second = Views([MeanView("A", "==", 1.0 / 11.0)])
    md = _market_data().with_views(
        (datetime(2024, 1, 2), first),
        (datetime(2024, 1, 3), second),
    )

    viewed = md.viewed_returns()

    assert [item.as_of for item in viewed] == [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]
    assert viewed[0].panel.dates.to_list() == [datetime(2024, 1, 2)]
    assert viewed[1].panel.dates.to_list() == [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]


def test_viewed_returns_at_uses_latest_registered_view_causally() -> None:
    md = _market_data().with_views(
        (datetime(2024, 1, 2), Views([MeanView("A", "==", 0.1)])),
        (datetime(2024, 1, 3), Views([MeanView("A", "==", 1.0 / 11.0)])),
    )

    viewed = md.viewed_returns(t=datetime(2024, 1, 3))

    assert viewed.as_of == datetime(2024, 1, 3)
    assert viewed.panel.dates.to_list() == [datetime(2024, 1, 2), datetime(2024, 1, 3)]


def test_viewed_returns_at_raises_without_registered_view() -> None:
    md = _market_data()

    with pytest.raises(ValueError, match="No views registered"):
        md.viewed_returns(t=datetime(2024, 1, 3))


def test_views_reject_non_return_panel() -> None:
    panel = ScenarioPanel.from_prices(_price_frame(), prob=np.array([0.2, 0.3, 0.5]))
    views = Views([MeanView("A", "==", 0.1)])

    with pytest.raises(ValueError, match="simple-return panel"):
        views.apply(panel)


# ---------------------------------------------------------------------------
# MarketData – cash_rate_asof
# ---------------------------------------------------------------------------


def test_cash_rate_asof_no_cash_returns_zero() -> None:
    md = MarketData.from_prices(_price_frame(), _universe())
    assert md.cash_rate_asof(datetime(2024, 1, 2)) == 0.0


def test_cash_rate_asof_returns_per_step_return() -> None:
    md = _market_data()
    rate = md.cash_rate_asof(datetime(2024, 1, 2))
    # latest known at 2024-01-02: 5.25% / 100 = 0.0525 annual
    expected = 0.0525 / md.config.cash_day_count
    assert rate == pytest.approx(expected)


def test_cash_rate_asof_accepts_decimal_rate_unit() -> None:
    cash = pl.DataFrame({"date": [datetime(2024, 1, 1)], "DFF": [0.0525]})
    md = MarketData.from_prices(
        _price_frame(),
        _universe(),
        cash=cash,
        config=MarketDataConfig(cash_rate_unit="decimal"),
    )

    assert md.cash_rate_asof(datetime(2024, 1, 2)) == pytest.approx(
        0.0525 / md.config.cash_day_count
    )


def test_cash_rate_asof_raises_on_no_prior_rate() -> None:
    md = _market_data()
    with pytest.raises(ValueError, match="No cash rate"):
        md.cash_rate_asof(datetime(2020, 1, 1))


def test_cash_rate_asof_with_step_size() -> None:
    md = _market_data()
    one_step = md.cash_rate_asof(datetime(2024, 1, 2), step_size=1)
    five_step = md.cash_rate_asof(datetime(2024, 1, 2), step_size=5)
    expected_five = 0.0525 * 5.0 / md.config.cash_day_count
    assert five_step == pytest.approx(expected_five)
    assert five_step > one_step


# ---------------------------------------------------------------------------
# MarketData – realised_cash_return
# ---------------------------------------------------------------------------


def test_realised_cash_return_no_cash_returns_zero() -> None:
    md = MarketData.from_prices(_price_frame(), _universe())
    r = md.realised_cash_return(datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert r == 0.0


def test_realised_cash_return_uses_optimizer_cash_convention() -> None:
    md = _market_data()
    r = md.realised_cash_return(datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert r == pytest.approx(md.cash_rate_asof(datetime(2024, 1, 1)))


def test_realised_cash_return_accepts_string_dates() -> None:
    md = _market_data()
    r = md.realised_cash_return("2024-01-01", "2024-01-02")
    assert r == pytest.approx(md.cash_rate_asof("2024-01-01"))


def test_realised_cash_return_multi_day_interval() -> None:
    md = _market_data()
    r = md.realised_cash_return(datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert r == pytest.approx(md.cash_rate_asof(datetime(2024, 1, 1), step_size=2))


def test_realised_cash_return_raises_on_no_prior_rate() -> None:
    md = _market_data()
    with pytest.raises(ValueError, match="No cash rate"):
        md.realised_cash_return(datetime(2020, 1, 1), datetime(2024, 1, 2))


# ---------------------------------------------------------------------------
# MarketData – snapshot_at / realised_step
# ---------------------------------------------------------------------------


def test_snapshot_at() -> None:
    md = _market_data()
    snap = md.snapshot_at(datetime(2024, 1, 2), datetime(2024, 1, 3))
    assert isinstance(snap, MarketSnapshot)
    assert snap.t == datetime(2024, 1, 2)
    assert snap.t_next == datetime(2024, 1, 3)
    assert snap.universe.assets == ["A", "B"]
    assert isinstance(snap.history, ScenarioPanel)
    np.testing.assert_allclose(snap.prices_t, [11.0, 22.0])
    assert snap.cash_rate > 0


def test_snapshot_at_accepts_string_dates() -> None:
    md = _market_data()
    snap = md.snapshot_at("2024-01-02", "2024-01-03")
    assert snap.t == datetime(2024, 1, 2)
    assert snap.t_next == datetime(2024, 1, 3)
    np.testing.assert_allclose(snap.prices_t, [11.0, 22.0])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_universe_with_factors_are_excluded_from_prices_at() -> None:
    universe = AssetUniverse(assets=["A"], factors=["B"])
    md = MarketData.from_prices(_price_frame(), universe)
    np.testing.assert_allclose(md.prices_at(datetime(2024, 1, 2)), [11.0])
    np.testing.assert_allclose(md.prices_at(datetime(2024, 1, 3)), [12.0])


def test_prices_at_returns_stored_raw_prices() -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)],
            "A": [15.123456789],
            "B": [30.987654321],
        }
    )
    md = MarketData.from_prices(
        df, _universe(), config=MarketDataConfig(periods_per_year=252)
    )
    np.testing.assert_array_equal(
        md.prices_at(datetime(2024, 1, 1)), [15.123456789, 30.987654321]
    )


def test_from_prices_accepts_string_date_columns() -> None:
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "A": [10.0, 11.0],
            "B": [20.0, 22.0],
        }
    )
    cash = pl.DataFrame({"date": ["2024-01-01"], "DFF": [5.0]})
    md = MarketData.from_prices(df, _universe(), cash=cash)
    assert md.trading_bars == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    np.testing.assert_allclose(md.prices_at("2024-01-02"), [11.0, 22.0])
    assert md.cash_rate_asof("2024-01-02") == pytest.approx(
        0.05 / md.config.cash_day_count
    )


def test_from_prices_rejects_non_iso_string_dates() -> None:
    df = pl.DataFrame(
        {
            "date": ["01-02-2024", "2024-01-02"],
            "A": [10.0, 11.0],
            "B": [20.0, 22.0],
        }
    )

    with pytest.raises(ValueError, match="ISO-8601"):
        MarketData.from_prices(df, _universe())


def test_cash_rate_asof_uses_latest_rate_before_t() -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 5)],
            "DFF": [5.0, 6.0],
        }
    )
    md = MarketData.from_prices(_price_frame(), _universe(), cash=df)
    # At t=2024-01-03, latest rate <= t is 5.0 (from 2024-01-01)
    assert md.cash_rate_asof(datetime(2024, 1, 3)) == pytest.approx(
        0.05 / md.config.cash_day_count
    )


def test_realised_cash_return_uses_rate_at_t_prev() -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "DFF": [5.0, 6.0],
        }
    )
    md = MarketData.from_prices(_price_frame(), _universe(), cash=df)
    # uses rate at 2024-01-01 (t_prev), which is 5.0%
    r = md.realised_cash_return(datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert r == pytest.approx(md.cash_rate_asof(datetime(2024, 1, 1), step_size=2))
