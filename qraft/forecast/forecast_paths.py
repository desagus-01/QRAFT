from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame

from qraft.core.panel import DatetimeSeries, ScenarioPanel, normalize_datetime_series
from qraft.core.probability.prob_vector import ProbVector, as_prob_vector
from qraft.core.universe import AssetSubset, AssetUniverse
from qraft.utils.visuals import plot_simulation_results

if TYPE_CHECKING:
    from qraft.forecast.pipelines.fitted_universe import FittedUniverse

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InnovationPaths:
    values: NDArray[np.floating]
    path_probs: ProbVector

    def __post_init__(self) -> None:
        if self.values.ndim < 1:
            raise ValueError("InnovationPaths.values must have at least one dimension")
        path_probs = as_prob_vector(self.path_probs, length=self.values.shape[0])
        object.__setattr__(self, "path_probs", path_probs)


@dataclass(frozen=True, slots=True)
class ForecastPaths:
    asset_paths: dict[str, NDArray[np.floating]]
    dates: DatetimeSeries
    path_probs: ProbVector
    initial_prices: dict[str, float]
    universe: AssetUniverse | None = None
    diagnostics: FittedUniverse | None = None

    def __post_init__(self) -> None:
        if not self.asset_paths:
            raise ValueError("ForecastPaths.asset_paths cannot be empty")

        base_shape: tuple[int, int] | None = None

        for asset, path in self.asset_paths.items():
            path = np.asarray(path, dtype=float)
            if path.ndim != 2:
                raise ValueError(
                    f"Forecast for {asset} must have shape "
                    f"(n_paths, n_horizons); got {path.shape}"
                )
            if not np.all(np.isfinite(path)):
                raise ValueError(f"Forecast for {asset} contains NaN or inf")
            if np.any(path <= 0):
                raise ValueError(
                    f"Forecast prices for {asset} must be strictly positive"
                )

            if base_shape is None:
                base_shape = path.shape
            elif path.shape != base_shape:
                raise ValueError(
                    f"Forecast shape mismatch for {asset}: {path.shape} != {base_shape}"
                )

            self.asset_paths[asset] = path

        assert base_shape is not None
        n_paths, n_horizons = base_shape

        path_probs = as_prob_vector(self.path_probs, length=n_paths)
        object.__setattr__(self, "path_probs", path_probs)
        if len(self.dates) != n_horizons:
            raise ValueError(
                f"dates length {len(self.dates)} must equal number of horizons "
                f"{n_horizons}"
            )
        object.__setattr__(self, "dates", normalize_datetime_series(self.dates))
        for asset, price in self.initial_prices.items():
            if not np.isfinite(price):
                raise ValueError(f"initial price for {asset} must be finite")
            if price <= 0:
                raise ValueError(f"initial price for {asset} must be strictly positive")

    @property
    def price_stack(self) -> NDArray[np.floating]:
        return self.price_stack_for(list(self.tradable_paths.keys()))

    @property
    def n_simulations(self) -> int:
        first = next(iter(self.asset_paths.values()))
        return first.shape[0]

    @property
    def n_horizons(self) -> int:
        first = next(iter(self.asset_paths.values()))
        return first.shape[1]

    @property
    def horizon_labels(self) -> list[str]:
        return [f"t+{step}" for step in range(1, self.n_horizons + 1)]

    @property
    def assets_and_factors_names(self) -> list[str]:
        return list(self.asset_paths.keys())

    @property
    def tradable_paths(self) -> dict[str, NDArray[np.floating]]:
        if self.universe is None:
            return self.asset_paths
        assets_set = set(self.universe.assets)
        return {k: v for k, v in self.asset_paths.items() if k in assets_set}

    @property
    def factor_paths(self) -> dict[str, NDArray[np.floating]]:
        if self.universe is None:
            return {}
        factors_set = set(self.universe.factors)
        return {k: v for k, v in self.asset_paths.items() if k in factors_set}

    def price_stack_for(self, assets: list[str]) -> NDArray[np.floating]:
        missing = set(assets) - self.asset_paths.keys()
        if missing:
            raise ValueError(
                f"Assets not found in forecast: {missing}. "
                f"Forecast contains {len(self.asset_paths)} asset(s); "
                f"{len(assets)} requested."
            )
        return np.stack([self.asset_paths[a] for a in assets], axis=0)

    def _paths_for(self, subset: AssetSubset) -> dict[str, NDArray[np.floating]]:
        if subset == "all":
            return self.asset_paths
        if subset == "tradable":
            return self.tradable_paths
        if subset == "factors":
            return self.factor_paths
        raise ValueError(f"Unknown subset: {subset}")

    def plot_asset_paths(self) -> None:
        for asset, path in self.tradable_paths.items():
            plot_simulation_results(path, title=asset)

    def at_step(
        self,
        step: int,
        *,
        subset: AssetSubset = "all",
    ) -> ScenarioPanel:
        """
        Return a 2D scenario panel at one forecast step.

        step is 1-based: step=1 is the first simulated forecast step.
        t0 is excluded; ForecastPaths contains no t0 column.
        """
        if step < 1:
            raise ValueError("step must be >= 1")

        if step > self.n_horizons:
            raise ValueError(
                f"step={step} is out of range. Forecast has {self.n_horizons} steps."
            )

        idx = step - 1
        paths = self._paths_for(subset)

        if not paths:
            raise ValueError(f"No paths available for subset={subset!r}")

        values = DataFrame({asset: arr[:, idx] for asset, arr in paths.items()})
        date = self.dates[idx]

        return ScenarioPanel(
            values=values,
            dates=pl.Series("date", [date] * values.height),
            prob=self.path_probs,
            kind="level",
        )
