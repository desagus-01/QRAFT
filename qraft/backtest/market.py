from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.distributions import state_smooth_probs, uniform_probs
from qraft.core.probability.prob_vector import ProbVector
from qraft.forecast.forecast_paths import AssetUniverse
from qraft.utils.helpers import str_to_datetime

PriceKind = Literal["price", "log_price"]
WeightingScheme = Literal["uniform", "state_smooth"]
DateLike = datetime | str


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Decision view at t — only information available at or before t."""

    t: datetime
    t_next: datetime
    assets: list[str]
    history: ScenarioPanel
    prices_t: NDArray[np.floating]
    cash_rate: float


@dataclass(frozen=True, slots=True)
class RealisedStep:
    """Realised leg over (t, t_next] — handed only to accounting / fill."""

    t: datetime
    t_next: datetime
    prices_next: NDArray[np.floating]
    cash_return: float


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    cash_column: str = "DFF"
    periods_per_year: int = 252
    cash_day_count: int = 360  # ACT/360 fed-funds accrual


@dataclass(frozen=True, slots=True)
class WindowWeighting:
    scheme: WeightingScheme = "uniform"
    half_life: float | None = None

    def __post_init__(self) -> None:
        if self.scheme == "state_smooth" and self.half_life is None:
            raise ValueError(
                "WindowWeighting(scheme='state_smooth') requires a half_life"
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
    price_kind: PriceKind
    universe: AssetUniverse
    cash: pl.DataFrame | None
    config: MarketDataConfig
    weighting: WindowWeighting

    @classmethod
    def from_prices(
        cls,
        data: pl.DataFrame,
        universe: AssetUniverse,
        *,
        cash: pl.DataFrame | None = None,
        weighting: WindowWeighting = WindowWeighting(),
        config: MarketDataConfig = MarketDataConfig(),
    ) -> "MarketData":
        return cls._from_frame(data, universe, "price", cash, weighting, config)

    @classmethod
    def from_log_prices(
        cls,
        data: pl.DataFrame,
        universe: AssetUniverse,
        *,
        cash: pl.DataFrame | None = None,
        weighting: WindowWeighting = WindowWeighting(),
        config: MarketDataConfig = MarketDataConfig(),
    ) -> "MarketData":
        return cls._from_frame(data, universe, "log_price", cash, weighting, config)

    @classmethod
    def _from_frame(
        cls,
        data: pl.DataFrame,
        universe: AssetUniverse,
        price_kind: PriceKind,
        cash: pl.DataFrame | None,
        weighting: WindowWeighting,
        config: MarketDataConfig,
    ) -> "MarketData":
        if "date" not in data.columns:
            raise ValueError("MarketData input must contain a 'date' column")
        frame = (
            cls._normalize_date_column(data)
            .select("date", *universe.all_tickers)
            .sort("date")
        )
        cash_sorted = (
            cls._normalize_date_column(cash).sort("date") if cash is not None else None
        )
        return cls(
            frame=frame,
            price_kind=price_kind,
            universe=universe,
            cash=cash_sorted,
            config=config,
            weighting=weighting,
        )

    @property
    def trading_bars(self) -> list[datetime]:
        return self.frame.get_column("date").to_list()

    def prices_at(self, t: DateLike) -> NDArray[np.floating]:
        t = self._as_datetime(t)
        row = self.frame.filter(pl.col("date") == t)
        if row.height == 0:
            raise ValueError(f"No market data on {t!r}")
        values = row.select(self.universe.assets).to_numpy().ravel()
        return self._to_prices(values, self.price_kind)

    def history_through(self, t: DateLike) -> ScenarioPanel:
        """Causal log-price window: rows with date <= t, weights from the window only."""
        t = self._as_datetime(t)
        window = self.frame.filter(pl.col("date") <= t)
        if window.height == 0:
            raise ValueError(f"No history on or before {t!r}")
        prob = self.weighting.probs(window.height)
        if self.price_kind == "log_price":
            return ScenarioPanel.from_log_prices(window, prob=prob)
        return ScenarioPanel.from_prices(window, prob=prob)

    def cash_rate_asof(self, t: DateLike, *, step_size: int = 1) -> float:
        """Ex-ante assumption: latest rate known at t, as a per-step return."""
        t = self._as_datetime(t)
        if self.cash is None:
            return 0.0
        prior = self.cash.filter(pl.col("date") <= t)
        if prior.height == 0:
            raise ValueError(f"No cash rate on or before {t!r}")
        annual = float(prior.get_column(self.config.cash_column)[-1]) / 100.0
        return (1.0 + annual) ** (step_size / self.config.periods_per_year) - 1.0

    def realised_cash_return(self, t_prev: DateLike, t: DateLike) -> float:
        """Realised accrual over (t_prev, t] on the rate entering the interval (ACT/360)."""
        t_prev = self._as_datetime(t_prev)
        t = self._as_datetime(t)
        if self.cash is None:
            return 0.0
        prior = self.cash.filter(pl.col("date") <= t_prev)
        if prior.height == 0:
            raise ValueError(f"No cash rate on or before {t_prev!r}")
        annual = float(prior.get_column(self.config.cash_column)[-1]) / 100.0
        return annual * (t - t_prev).days / self.config.cash_day_count

    def snapshot_at(
        self, t: DateLike, t_next: DateLike, *, step_size: int = 1
    ) -> MarketSnapshot:
        t = self._as_datetime(t)
        t_next = self._as_datetime(t_next)
        return MarketSnapshot(
            t=t,
            t_next=t_next,
            assets=list(self.universe.assets),
            history=self.history_through(t),
            prices_t=self.prices_at(t),
            cash_rate=self.cash_rate_asof(t, step_size=step_size),
        )

    def realised_step(self, t: DateLike, t_next: DateLike) -> RealisedStep:
        t = self._as_datetime(t)
        t_next = self._as_datetime(t_next)
        return RealisedStep(
            t=t,
            t_next=t_next,
            prices_next=self.prices_at(t_next),
            cash_return=self.realised_cash_return(t, t_next),
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

    @staticmethod
    def _to_prices(
        values: NDArray[np.floating], price_kind: PriceKind
    ) -> NDArray[np.floating]:
        if price_kind == "log_price":
            return np.exp(values)
        return np.asarray(values, dtype=float)
