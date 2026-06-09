import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from qraft.core.estimation import (
    weighted_correlation,
    weighted_covariance,
    weighted_mean,
)
from qraft.forecast.forecast_paths import AssetSubset, ForecastPaths
from qraft.core.probability.prob_vector import ProbVector
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
PnL_OPTIONS = Literal["relative", "absolute", "log"]


def pnl_from_values(
    values: NDArray[np.floating],
    mode: PnL_OPTIONS = "relative",
) -> NDArray[np.floating]:
    if mode not in {"relative", "absolute", "log"}:
        raise ValueError(f"Unknown mode: {mode!r}")

    if values.ndim != 2:
        raise ValueError(
            f"values must be 2-D (n_paths, n_periods); got shape {values.shape}"
        )

    prev = values[:, :-1]
    curr = values[:, 1:]

    if mode == "absolute":
        return curr - prev

    if mode == "relative":
        denom = prev.astype(float, copy=True)
        return (curr - prev) / denom

    denom = prev.astype(float, copy=True)
    ratio = curr / denom
    return np.log(ratio)


def incremental_returns_for_asset(
    price_paths: NDArray[np.floating],
    initial_price: float,
    horizons: int,
    pnl_type: PnL_OPTIONS = "relative",
) -> NDArray[np.floating]:
    n_paths = price_paths.shape[0]
    t0_col = np.full((n_paths, 1), initial_price, dtype=float)
    values = np.concatenate([t0_col, price_paths], axis=1)
    inc = pnl_from_values(values, mode=pnl_type)
    return inc[:, :horizons]


def incremental_returns_from_forecast_paths(
    forecast_paths: ForecastPaths,
    horizons: int | None = None,
    subset: AssetSubset = "tradable",
    pnl_type: PnL_OPTIONS = "relative",
) -> dict[str, NDArray[np.floating]]:
    paths = forecast_paths._paths_for(subset)
    if not paths:
        raise ValueError(f"No paths available for subset={subset!r}")

    max_horizons = forecast_paths.n_horizons
    if horizons is None:
        horizons = max_horizons
    if horizons > max_horizons:
        raise ValueError(
            f"horizons={horizons} exceeds the number of available forecast "
            f"steps ({max_horizons})."
        )

    initial_prices = forecast_paths.initial_prices

    return {
        asset: incremental_returns_for_asset(
            price_paths=price_paths,
            initial_price=initial_prices[asset],
            horizons=horizons,
            pnl_type=pnl_type,
        )
        for asset, price_paths in paths.items()
    }


@dataclass(frozen=True, slots=True)
class HorizonMoments:
    assets: list[str]
    correlations: NDArray[np.floating]
    covariances: NDArray[np.floating]
    mean: NDArray[np.floating]
    scenario_returns: NDArray[np.floating]
    scenario_probs: ProbVector
    cash_return: NDArray[np.floating]

    def __post_init__(self) -> None:
        n_h, n_a = self.mean.shape
        if self.scenario_returns.shape[1:] != (n_h, n_a):
            raise ValueError(
                f"scenario_returns shape {self.scenario_returns.shape} "
                f"inconsistent with mean shape {self.mean.shape}"
            )
        if self.scenario_returns.shape[0] != len(self.scenario_probs):
            raise ValueError(
                f"scenario_returns has {self.scenario_returns.shape[0]} paths "
                f"but scenario_probs has length {len(self.scenario_probs)}"
            )

    @staticmethod
    def get_cash_return(
        path: str,
        step_size: int,
        periods_per_year: int = 252,
    ) -> NDArray[np.floating]:
        cash_return = pl.read_csv(path, try_parse_dates=True)
        annual_rate = (
            cash_return.filter(pl.col("date") == pl.col("date").max())
            .drop("date")
            .to_numpy()
            .ravel()
        ) / 100
        # Convert: (1 + r_annual)^(step/252) - 1
        return (1.0 + annual_rate) ** (step_size / periods_per_year) - 1.0

    @property
    def n_horizons(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.mean.shape[1])

    @property
    def n_scenarios(self) -> int:
        return int(self.scenario_returns.shape[0])

    @property
    def mean_frame(self) -> pl.DataFrame:
        """Wide frame: one row per horizon, one column per asset."""
        rows = [
            {
                "horizon": h + 1,
                **{a: float(self.mean[h, i]) for i, a in enumerate(self.assets)},
            }
            for h in range(self.n_horizons)
        ]
        return pl.DataFrame(rows)

    def _matrix_frame(self, matrix: NDArray[np.floating], horizon: int) -> pl.DataFrame:
        """Return slice [horizon] of an (H, N, N) array as a square DataFrame."""
        m = matrix[horizon]
        data: dict[str, list] = {"asset": self.assets}
        for i, asset in enumerate(self.assets):
            data[asset] = m[:, i].tolist()
        return pl.DataFrame(data)

    def correlation_frame(self, horizon: int = 0) -> pl.DataFrame:
        return self._matrix_frame(self.correlations, horizon)

    def covariance_frame(self, horizon: int = 0) -> pl.DataFrame:
        return self._matrix_frame(self.covariances, horizon)

    @staticmethod
    def _drop_assets_by_expectation_tolerance(
        assets: list[str],
        means: NDArray[np.floating],
        covariances: NDArray[np.floating],
        correlations: NDArray[np.floating],
        scenario_returns: NDArray[np.floating],
        expectation_tolerance: float | None,
    ) -> tuple[
        list[str],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        if expectation_tolerance is None:
            return assets, means, covariances, correlations, scenario_returns

        if expectation_tolerance < 0:
            raise ValueError("expectation_tolerance must be non-negative.")

        breached = np.abs(means) > expectation_tolerance
        drop_mask = np.any(breached, axis=0)

        if not np.any(drop_mask):
            logger.info(
                "No assets dropped. All expected returns are within ±%.6e.",
                expectation_tolerance,
            )
            return assets, means, covariances, correlations, scenario_returns

        keep_idx = np.where(~drop_mask)[0]
        drop_idx = np.where(drop_mask)[0]

        if len(keep_idx) == 0:
            raise ValueError(
                "All assets were dropped because their expected returns exceeded "
                f"±{expectation_tolerance}."
            )

        for asset_idx in drop_idx:
            asset = assets[asset_idx]
            bad_horizons = np.where(breached[:, asset_idx])[0]
            offending_values = ", ".join(
                f"horizon {h + 1}: mean={means[h, asset_idx]:.6e}" for h in bad_horizons
            )
            logger.warning(
                "Dropping asset %s because expected return breached ±%.6e. "
                "Offending values: %s",
                asset,
                expectation_tolerance,
                offending_values,
            )

        filtered_assets = [assets[i] for i in keep_idx]
        filtered_means = means[:, keep_idx]
        filtered_covariances = covariances[:, keep_idx][:, :, keep_idx]
        filtered_correlations = correlations[:, keep_idx][:, :, keep_idx]
        filtered_scenario_returns = scenario_returns[:, :, keep_idx]

        logger.warning(
            "Dropped %d asset(s) due to expectation_tolerance=%.6e. "
            "Remaining assets: %d.",
            len(drop_idx),
            expectation_tolerance,
            len(filtered_assets),
        )

        return (
            filtered_assets,
            filtered_means,
            filtered_covariances,
            filtered_correlations,
            filtered_scenario_returns,
        )

    @classmethod
    def from_forecast_paths(
        cls,
        forecast_paths: ForecastPaths,
        cash_path: str,
        horizons: int | None = None,
        subset: AssetSubset = "tradable",
        pnl_type: PnL_OPTIONS = "relative",
        expectation_tolerance: float | None = 1.0,
        step_size: int = 1,
        periods_per_year: int = 252,
    ) -> "HorizonMoments":
        """
        Build stacked multi-horizon moments from simulated price paths.

        Moments at each horizon h use the incremental period-over-period
        return from t_{h-1} to t_h, not the cumulative return from t_0.
        """
        pnl_by_asset = incremental_returns_from_forecast_paths(
            forecast_paths=forecast_paths,
            horizons=horizons,
            subset=subset,
            pnl_type=pnl_type,
        )

        assets = list(pnl_by_asset.keys())
        n_assets = len(assets)
        horizons = next(iter(pnl_by_asset.values())).shape[1]
        prob = forecast_paths.path_probs
        n_paths = forecast_paths.n_simulations

        inc_returns = np.empty((n_paths, horizons, n_assets), dtype=float)
        for col_idx, (asset, asset_returns) in enumerate(pnl_by_asset.items()):
            inc_returns[:, :, col_idx] = asset_returns

        means = np.empty((horizons, n_assets), dtype=float)
        covs = np.empty((horizons, n_assets, n_assets), dtype=float)
        corrs = np.empty((horizons, n_assets, n_assets), dtype=float)

        for h in range(horizons):
            pnl_matrix = inc_returns[:, h, :]
            means[h] = weighted_mean(data=pnl_matrix, prob=prob)
            covs[h] = weighted_covariance(data=pnl_matrix, prob=prob)
            corrs[h] = weighted_correlation(data=pnl_matrix, prob=prob)

        assets, means, covs, corrs, inc_returns = (
            cls._drop_assets_by_expectation_tolerance(
                assets=assets,
                means=means,
                covariances=covs,
                correlations=corrs,
                expectation_tolerance=expectation_tolerance,
                scenario_returns=inc_returns,
            )
        )

        return cls(
            assets=assets,
            correlations=corrs,
            covariances=covs,
            mean=means,
            scenario_returns=inc_returns,
            scenario_probs=prob,
            cash_return=cls.get_cash_return(
                path=cash_path, step_size=step_size, periods_per_year=periods_per_year
            ),
        )


@dataclass(frozen=True, slots=True)
class MomentsConfig:
    cash_path: str
    horizons: int | None = None
    subset: AssetSubset = "tradable"
    pnl_type: PnL_OPTIONS = "relative"
    expectation_tolerance: float | None = 1.0
    step_size: int = 1
    periods_per_year: int = 252
