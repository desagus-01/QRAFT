from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.construction.market_snapshot import MarketSnapshot
from qraft.core.cadence import resolve_periods_per_year
from qraft.core.panel import ScenarioPanel
from qraft.core.probability.distributions import state_smooth_probs, uniform_probs
from qraft.core.probability.prob_vector import ProbVector
from qraft.forecast.forecast_paths import AssetUniverse
from qraft.utils.helpers import str_to_datetime

PriceKind = Literal["price", "log_price"]
WeightingScheme = Literal["uniform", "state_smooth"]
DateLike = datetime | str


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    cash_column: str = "DFF"
    periods_per_year: float | None = None
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
        values = frame.select(*universe.all_tickers).to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError("MarketData prices must contain only finite values")
        if price_kind == "price" and np.any(values <= 0):
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
            periods_per_year=resolve_periods_per_year(
                frame.get_column("date").to_list(), config.periods_per_year
            ),
            cash_day_count=config.cash_day_count,
        )
        return cls(
            frame=frame,
            price_kind=price_kind,
            universe=universe,
            cash=cash_sorted,
            config=resolved_config,
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
        """Ex-ante assumption: latest rate known at t, as an ACT/day-count return."""
        t = self._as_datetime(t)
        if self.cash is None:
            return 0.0
        prior = self.cash.filter(pl.col("date") <= t)
        if prior.height == 0:
            raise ValueError(f"No cash rate on or before {t!r}")
        annual = float(prior.get_column(self.config.cash_column)[-1]) / 100.0
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

    @staticmethod
    def _to_prices(
        values: NDArray[np.floating], price_kind: PriceKind
    ) -> NDArray[np.floating]:
        if price_kind == "log_price":
            return np.exp(values)
        return np.asarray(values, dtype=float)
