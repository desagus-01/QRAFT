from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.core.cadence import resolve_periods_per_year
from qraft.core.panel import ScenarioPanel, expand_posterior_to_parent
from qraft.core.probability.distributions import state_smooth_probs, uniform_probs
from qraft.core.probability.prob_vector import ProbVector
from qraft.core.scenarios.transforms import ViewedDistribution
from qraft.core.scenarios.views import (
    ScenarioView,
    ViewInput,
    ViewState,
    normalize_view_event,
)
from qraft.core.snapshot import MarketSnapshot
from qraft.core.universe import AssetUniverse
from qraft.utils.helpers import str_to_datetime
from qraft.utils.log import info_event

WeightingScheme = Literal["uniform", "state_smooth"]
CashRateUnit = Literal["percent", "decimal"]
DateLike = datetime | str
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    cash_column: str = "DFF"
    cash_rate_unit: CashRateUnit = "percent"
    periods_per_year: float | None = None
    cash_day_count: int = 360  # ACT/360 fed-funds accrual


@dataclass(frozen=True, slots=True)
class HistoryWeighting:
    scheme: WeightingScheme = "uniform"
    half_life: float | None = None

    def __post_init__(self) -> None:
        if self.scheme == "state_smooth" and self.half_life is None:
            raise ValueError(
                "HistoryWeighting(scheme='state_smooth') requires a half_life"
            )

    def probs(self, n: int) -> ProbVector:
        if self.scheme == "uniform":
            return uniform_probs(n)
        if self.half_life is not None:
            return state_smooth_probs(n, half_life=self.half_life)
        raise ValueError("half life must be inputted to use state_smooth_probs")


@dataclass(frozen=True, slots=True)
class MarketData:
    frame: pl.DataFrame
    universe: AssetUniverse
    cash: pl.DataFrame | None
    config: MarketDataConfig
    history_weighting: HistoryWeighting
    views: ViewState = ViewState()

    @classmethod
    def from_prices(
        cls,
        data: pl.DataFrame,
        universe: AssetUniverse,
        *,
        cash: pl.DataFrame | None = None,
        history_weighting: HistoryWeighting = HistoryWeighting(),
        config: MarketDataConfig = MarketDataConfig(),
    ) -> "MarketData":
        return cls._from_frame(data, universe, cash, history_weighting, config)

    @classmethod
    def _from_frame(
        cls,
        data: pl.DataFrame,
        universe: AssetUniverse,
        cash: pl.DataFrame | None,
        history_weighting: HistoryWeighting,
        config: MarketDataConfig,
    ) -> "MarketData":
        if "date" not in data.columns:
            raise ValueError("MarketData input must contain a 'date' column")
        frame = (
            cls._normalize_date_column(data)
            .select("date", *universe.all_tickers)
            .sort("date")
        )
        values = frame.select(*universe.all_tickers).to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError("MarketData prices must contain only finite values")
        if np.any(values <= 0):
            raise ValueError("MarketData prices must be strictly positive")
        cash_sorted = (
            cls._normalize_date_column(cash).sort("date") if cash is not None else None
        )
        if cash_sorted is not None:
            cash_values = cash_sorted.get_column(config.cash_column).to_numpy()
            if not np.all(np.isfinite(cash_values)):
                raise ValueError("Cash rates must contain only finite values")
        resolved_config = MarketDataConfig(
            cash_column=config.cash_column,
            cash_rate_unit=config.cash_rate_unit,
            periods_per_year=resolve_periods_per_year(
                frame.get_column("date").to_list(), config.periods_per_year
            ),
            cash_day_count=config.cash_day_count,
        )
        return cls(
            frame=frame,
            universe=universe,
            cash=cash_sorted,
            config=resolved_config,
            history_weighting=history_weighting,
        )

    def with_views(self, *events: ViewInput) -> "MarketData":
        normalized = tuple(
            sorted((normalize_view_event(e) for e in events), key=lambda e: e.as_of)
        )
        return replace(self, views=ViewState(events=normalized))

    @property
    def trading_bars(self) -> list[datetime]:
        return self.frame.get_column("date").to_list()

    def prices_at(self, t: DateLike) -> NDArray[np.floating]:
        t = self._as_datetime(t)
        row = self.frame.filter(pl.col("date") == t)
        if row.height == 0:
            raise ValueError(f"No market data on {t!r}")
        values = row.select(self.universe.assets).to_numpy().ravel()
        return np.asarray(values, dtype=float)

    def history_through(self, t: DateLike) -> ScenarioPanel:
        """Causal log-price window: rows with date <= t, weights from the window only."""
        t = self._as_datetime(t)
        panel = self._log_price_history_through(t)
        event = self.views.latest_event_at(t)
        if event is None:
            return panel
        returns = self._simple_return_history_through(t)
        self._log_views_applied(
            target="history_through",
            requested_bar=t,
            view_as_of=event.as_of,
            views=event.views,
            panel=returns,
        )
        if hasattr(event.views, "view_distribution"):
            viewed = event.views.view_distribution(returns, as_of=t)
            return panel.with_prob(viewed.prob_for(panel))
        viewed_panel = event.views.apply(returns)
        return panel.with_prob(
            expand_posterior_to_parent(viewed_panel.prob, panel.prob, 1)
        )

    @overload
    def viewed_returns(
        self, views: None = None, t: None = None
    ) -> list[ViewedDistribution]: ...

    @overload
    def viewed_returns(
        self, views: None = None, t: DateLike = ...
    ) -> ViewedDistribution: ...

    @overload
    def viewed_returns(
        self, views: ScenarioView = ..., t: DateLike | None = None
    ) -> ViewedDistribution: ...

    def viewed_returns(
        self, views: ScenarioView | None = None, t: DateLike | None = None
    ) -> ViewedDistribution | list[ViewedDistribution]:
        if views is not None:
            if not hasattr(views, "view_distribution"):
                raise TypeError("views must provide view_distribution(panel)")
            as_of = self.trading_bars[-1] if t is None else self._as_datetime(t)
            returns = self._simple_return_history_through(as_of)
            self._log_views_applied(
                target="viewed_returns",
                requested_bar=as_of,
                view_as_of=as_of,
                views=views,
                panel=returns,
            )
            return views.view_distribution(returns, as_of=as_of)
        if t is not None:
            as_of = self._as_datetime(t)
            event = self.views.latest_event_at(as_of)
            if event is None:
                raise ValueError(f"No views registered on or before {as_of!r}")
            returns = self._simple_return_history_through(as_of)
            self._log_views_applied(
                target="viewed_returns",
                requested_bar=as_of,
                view_as_of=event.as_of,
                views=event.views,
                panel=returns,
            )
            return event.views.view_distribution(returns, as_of=event.as_of)
        return [
            self._viewed_returns_for_event(event.views, event.as_of)
            for event in self.views.events
        ]

    def _viewed_returns_for_event(
        self, views: ScenarioView, as_of: datetime
    ) -> ViewedDistribution:
        returns = self._simple_return_history_through(as_of)
        self._log_views_applied(
            target="viewed_returns",
            requested_bar=as_of,
            view_as_of=as_of,
            views=views,
            panel=returns,
        )
        return views.view_distribution(returns, as_of=as_of)

    def _log_views_applied(
        self,
        *,
        target: str,
        requested_bar: datetime,
        view_as_of: datetime,
        views: ScenarioView,
        panel: ScenarioPanel,
    ) -> None:
        dates = panel.dates.to_list()
        info_event(
            logger,
            "views.applied",
            "Applying scenario views",
            target=target,
            requested_bar=requested_bar,
            view_as_of=view_as_of,
            view_type=type(views).__name__,
            panel_kind=panel.kind,
            bar_count=len(dates),
            first_bar=dates[0] if dates else None,
            last_bar=dates[-1] if dates else None,
        )

    def _raw_history_through(self, t: datetime) -> tuple[pl.DataFrame, ProbVector]:
        window = self.frame.filter(pl.col("date") <= t)
        if window.height == 0:
            raise ValueError(f"No history on or before {t!r}")
        prob = self.history_weighting.probs(window.height)
        return window, prob

    def _log_price_history_through(self, t: datetime) -> ScenarioPanel:
        window, prob = self._raw_history_through(t)
        return ScenarioPanel.from_prices(window, prob=prob)

    def _simple_return_history_through(self, t: datetime) -> ScenarioPanel:
        window, prob = self._raw_history_through(t)
        return ScenarioPanel.from_levels(window, prob=prob).to_simple_returns()

    def cash_rate_asof(self, t: DateLike, *, step_size: int = 1) -> float:
        """Ex-ante assumption: latest rate known at t, as an ACT/day-count return."""
        t = self._as_datetime(t)
        if self.cash is None:
            return 0.0
        prior = self.cash.filter(pl.col("date") <= t)
        if prior.height == 0:
            raise ValueError(f"No cash rate on or before {t!r}")
        raw_annual = float(prior.get_column(self.config.cash_column)[-1])
        if self.config.cash_rate_unit == "percent":
            annual = raw_annual / 100.0
        elif self.config.cash_rate_unit == "decimal":
            annual = raw_annual
        else:
            raise ValueError("cash_rate_unit must be 'percent' or 'decimal'")
        if not np.isfinite(annual):
            raise ValueError("Cash rate must be finite")
        return annual * step_size / self.config.cash_day_count

    def realised_cash_return(self, t_prev: DateLike, t: DateLike) -> float:
        """Realised accrual over the actual elapsed calendar days."""
        t_prev = self._as_datetime(t_prev)
        t = self._as_datetime(t)
        if self.cash is None:
            return 0.0
        days = (t - t_prev).total_seconds() / 86_400.0
        if days <= 0:
            raise ValueError("t must be after t_prev to realise cash return.")
        return self.cash_rate_asof(t_prev, step_size=days)

    def snapshot_at(
        self, t: DateLike, t_next: DateLike, *, step_size: int = 1
    ) -> MarketSnapshot:
        t = self._as_datetime(t)
        t_next = self._as_datetime(t_next)
        return MarketSnapshot(
            t=t,
            t_next=t_next,
            universe=self.universe,
            history=self.history_through(t),
            prices_t=self.prices_at(t),
            cash_rate=self.cash_rate_asof(t, step_size=step_size),
        )

    @staticmethod
    def _as_datetime(t: DateLike) -> datetime:
        if isinstance(t, str):
            return str_to_datetime(t)
        return t

    @staticmethod
    def _normalize_date_column(data: pl.DataFrame) -> pl.DataFrame:
        dtype = data.schema["date"]
        if dtype == pl.Date:
            return data.with_columns(pl.col("date").cast(pl.Datetime))
        if dtype == pl.String:
            return data.with_columns(
                pl.col("date")
                .map_elements(str_to_datetime, return_dtype=pl.Datetime)
                .alias("date")
            )
        return data
