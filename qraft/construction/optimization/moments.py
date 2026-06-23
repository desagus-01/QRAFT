import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.core.estimation import (
    weighted_correlation,
    weighted_covariance,
    weighted_mean,
)
from qraft.core.panel import ScenarioPanel
from qraft.core.probability.prob_vector import ProbVector, validate_prob_vector
from qraft.forecast.forecast_paths import AssetSubset, ForecastPaths

logger = logging.getLogger(__name__)
PnL_OPTIONS = Literal["relative", "absolute", "log"]
ExpectedReturnSource = Literal["historical", "forecast"]
RiskSource = Literal["covariance", "cvar", "both"]


def psd_sqrt_factor(covariance: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return F with ``F @ F.T ≈ covariance`` (negative eigenvalues clamped)."""
    cov = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals))


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
class PolicyInputs:
    """
    Optimizer-ready horizon-indexed policy inputs.
    """

    assets: list[str]
    mean: NDArray[np.floating] | None = None
    covariances: NDArray[np.floating] | None = None
    correlations: NDArray[np.floating] | None = None
    scenario_returns: NDArray[np.floating] | None = None
    scenario_probs: ProbVector | None = None
    cash_return: NDArray[np.floating] | None = None
    cov_factor: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("PolicyInputs.assets cannot be empty.")

        n_a = len(self.assets)
        horizon_counts: list[int] = []

        if self.mean is not None:
            self._validate_matrix("mean", self.mean, n_a)
            horizon_counts.append(int(self.mean.shape[0]))

        if self.covariances is not None:
            self._validate_cube("covariances", self.covariances, n_a)
            horizon_counts.append(int(self.covariances.shape[0]))

        if self.correlations is not None:
            self._validate_cube("correlations", self.correlations, n_a)
            horizon_counts.append(int(self.correlations.shape[0]))

        if self.cov_factor is not None:
            self._validate_cube("cov_factor", self.cov_factor, n_a)
            horizon_counts.append(int(self.cov_factor.shape[0]))

        if self.scenario_returns is not None:
            if self.scenario_returns.ndim != 3:
                raise ValueError(
                    "scenario_returns must have shape "
                    "(n_scenarios, n_horizons, n_assets)."
                )
            if self.scenario_returns.shape[2] != n_a:
                raise ValueError(
                    f"scenario_returns asset dimension {self.scenario_returns.shape[2]} "
                    f"does not match assets length {n_a}."
                )
            horizon_counts.append(int(self.scenario_returns.shape[1]))

        if self.scenario_probs is not None:
            validate_prob_vector(self.scenario_probs)

        if (self.scenario_returns is None) != (self.scenario_probs is None):
            raise ValueError(
                "scenario_returns and scenario_probs must be provided together."
            )

        if self.scenario_returns is not None and self.scenario_probs is not None:
            if self.scenario_returns.shape[0] != len(self.scenario_probs):
                raise ValueError(
                    f"scenario_returns has {self.scenario_returns.shape[0]} paths "
                    f"but scenario_probs has length {len(self.scenario_probs)}."
                )

        if self.cash_return is not None:
            cash = np.asarray(self.cash_return, dtype=float).ravel()
            if cash.size == 0:
                raise ValueError("cash_return cannot be empty.")
            if cash.size > 1:
                horizon_counts.append(cash.size)
            object.__setattr__(self, "cash_return", cash)

        if not horizon_counts:
            raise ValueError(
                "PolicyInputs must contain at least one horizon-indexed input."
            )

        first = horizon_counts[0]
        if any(h != first for h in horizon_counts):
            raise ValueError(
                f"All horizon-indexed inputs must have the same horizon count; "
                f"got {horizon_counts}."
            )

    @staticmethod
    def _validate_matrix(name: str, value: NDArray[np.floating], n_assets: int) -> None:
        if value.ndim != 2 or value.shape[1] != n_assets:
            raise ValueError(
                f"{name} must have shape (n_horizons, {n_assets}); got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or inf.")

    @staticmethod
    def _validate_cube(name: str, value: NDArray[np.floating], n_assets: int) -> None:
        if value.ndim != 3 or value.shape[1:] != (n_assets, n_assets):
            raise ValueError(
                f"{name} must have shape "
                f"(n_horizons, {n_assets}, {n_assets}); got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or inf.")

    def require_mean(self) -> NDArray[np.floating]:
        if self.mean is None:
            raise ValueError("This objective requires PolicyInputs.mean.")
        return self.mean

    def require_covariances(self) -> NDArray[np.floating]:
        if self.covariances is None:
            raise ValueError("This objective requires PolicyInputs.covariances.")
        return self.covariances

    def require_scenarios(
        self,
    ) -> tuple[NDArray[np.floating], ProbVector]:
        if self.scenario_returns is None or self.scenario_probs is None:
            raise ValueError(
                "This objective requires forecast-based scenario_returns and "
                "scenario_probs."
            )
        return self.scenario_returns, self.scenario_probs

    def require_cash_return(self) -> NDArray[np.floating]:
        if self.cash_return is None:
            raise ValueError("This objective requires PolicyInputs.cash_return.")
        return self.cash_return

    def astype(self, dtype: type) -> "PolicyInputs":
        """Down-cast the memory-heavy arrays (scenario tensor + covariance
        cubes) to ``dtype``. ``mean``/``cash_return``/``scenario_probs`` stay
        as-is (small, precision-sensitive). cvxpy upcasts at solve time."""

        def cast(a: NDArray[np.floating] | None) -> NDArray[np.floating] | None:
            return None if a is None else a.astype(dtype, copy=False)

        return replace(
            self,
            covariances=cast(self.covariances),
            correlations=cast(self.correlations),
            cov_factor=cast(self.cov_factor),
            scenario_returns=cast(self.scenario_returns),
        )

    @staticmethod
    def get_cash_return(
        path: str,
        step_size: int,
        periods_per_year: int = 252,
        *,
        as_of: datetime | None = None,
    ) -> NDArray[np.floating]:
        cash_return = pl.read_csv(path, try_parse_dates=True)
        if as_of is not None:
            cash_return = cash_return.filter(pl.col("date") <= as_of)
            if cash_return.height == 0:
                raise ValueError(f"No cash rate available on or before {as_of!r}.")
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
        for value in (
            self.mean,
            self.covariances,
            self.correlations,
            self.cov_factor,
        ):
            if value is not None:
                return int(value.shape[0])
        if self.scenario_returns is not None:
            return int(self.scenario_returns.shape[1])
        if self.cash_return is not None:
            return np.asarray(self.cash_return).size
        raise RuntimeError("PolicyInputs has no horizon-indexed input.")

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_scenarios(self) -> int:
        if self.scenario_returns is None:
            return 1
        return int(self.scenario_returns.shape[0])

    @property
    def mean_frame(self) -> pl.DataFrame:
        """Wide frame: one row per horizon, one column per asset."""
        mean = self.require_mean()
        rows = [
            {
                "horizon": h + 1,
                **{a: float(mean[h, i]) for i, a in enumerate(self.assets)},
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
        if self.correlations is None:
            raise ValueError("PolicyInputs.correlations is required.")
        return self._matrix_frame(self.correlations, horizon)

    def covariance_frame(self, horizon: int = 0) -> pl.DataFrame:
        return self._matrix_frame(self.require_covariances(), horizon)

    @staticmethod
    def _filter_by_asset_indices(
        value: NDArray[np.floating] | None,
        keep_idx: NDArray[np.integer],
        *,
        kind: Literal["matrix", "cube", "scenarios"],
    ) -> NDArray[np.floating] | None:
        if value is None:
            return None
        if kind == "matrix":
            return value[:, keep_idx]
        if kind == "cube":
            return value[:, keep_idx][:, :, keep_idx]
        if kind == "scenarios":
            return value[:, :, keep_idx]
        raise ValueError(f"Unknown filter kind: {kind!r}")

    @classmethod
    def _filter_policy_fields_by_expectation_tolerance(
        cls,
        *,
        assets: list[str],
        mean: NDArray[np.floating],
        covariances: NDArray[np.floating] | None,
        correlations: NDArray[np.floating] | None,
        scenario_returns: NDArray[np.floating] | None,
        cov_factor: NDArray[np.floating] | None,
        expectation_tolerance: float | None,
    ) -> tuple[
        list[str],
        NDArray[np.floating],
        NDArray[np.floating] | None,
        NDArray[np.floating] | None,
        NDArray[np.floating] | None,
        NDArray[np.floating] | None,
    ]:
        if expectation_tolerance is None:
            return (
                assets,
                mean,
                covariances,
                correlations,
                scenario_returns,
                cov_factor,
            )

        if expectation_tolerance < 0:
            raise ValueError("expectation_tolerance must be non-negative.")

        breached = np.abs(mean) > expectation_tolerance
        drop_mask = np.any(breached, axis=0)

        if not np.any(drop_mask):
            logger.info(
                "No assets dropped. All expected returns are within ±%.6e.",
                expectation_tolerance,
            )
            return (
                assets,
                mean,
                covariances,
                correlations,
                scenario_returns,
                cov_factor,
            )

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
                f"horizon {h + 1}: mean={mean[h, asset_idx]:.6e}" for h in bad_horizons
            )
            logger.warning(
                "Dropping asset %s because expected return breached ±%.6e. "
                "Offending values: %s",
                asset,
                expectation_tolerance,
                offending_values,
            )

        filtered_assets = [assets[i] for i in keep_idx]
        logger.warning(
            "Dropped %d asset(s) due to expectation_tolerance=%.6e. "
            "Remaining assets: %d.",
            len(drop_idx),
            expectation_tolerance,
            len(filtered_assets),
        )

        return (
            filtered_assets,
            mean[:, keep_idx],
            cls._filter_by_asset_indices(covariances, keep_idx, kind="cube"),
            cls._filter_by_asset_indices(correlations, keep_idx, kind="cube"),
            cls._filter_by_asset_indices(scenario_returns, keep_idx, kind="scenarios"),
            cls._filter_by_asset_indices(cov_factor, keep_idx, kind="cube"),
        )

    @staticmethod
    def _historical_mean(
        history: ScenarioPanel,
        *,
        assets: list[str],
        horizons: int,
        mean_decay: float,
    ) -> NDArray[np.floating]:
        if history.kind == "log_price":
            returns = history.diff().drop_nulls()
        elif history.kind == "return":
            returns = history.drop_nulls()
        else:
            raise ValueError(
                "historical expected returns require history.kind "
                "'log_price' or 'return'."
            )

        missing = set(assets) - set(returns.values.columns)
        if missing:
            raise ValueError(f"Assets not present in history: {sorted(missing)}.")

        data = returns.values.select(assets).to_numpy()
        base_mean = weighted_mean(data=data, prob=returns.prob)
        decay = np.asarray([mean_decay**h for h in range(horizons)], dtype=float)
        return decay[:, None] * base_mean[None, :]

    @classmethod
    def from_policy_sources(
        cls,
        *,
        forecasts: ForecastPaths,
        cash_path: str,
        expected_returns: ExpectedReturnSource = "forecast",
        risk: RiskSource = "both",
        history: ScenarioPanel | None = None,
        horizons: int | None = None,
        subset: AssetSubset = "tradable",
        pnl_type: PnL_OPTIONS = "relative",
        expectation_tolerance: float | None = 1.0,
        mean_decay: float = 1.0,
        step_size: int = 1,
        periods_per_year: int = 252,
        as_of: datetime | None = None,
    ) -> "PolicyInputs":
        """
        Build policy inputs from two simple choices: expected returns and risk.

        ``expected_returns`` selects the source of the optimizer mean. ``risk``
        always uses ``ForecastPaths`` and selects which risk representation is
        included.
        """
        if expected_returns not in {"historical", "forecast"}:
            raise ValueError(
                "expected_returns must be one of 'historical' or 'forecast'."
            )
        if risk not in {"covariance", "cvar", "both"}:
            raise ValueError("risk must be one of 'covariance', 'cvar', or 'both'.")
        if expected_returns == "historical" and history is None:
            raise ValueError("history is required when expected_returns='historical'.")

        pnl_by_asset = incremental_returns_from_forecast_paths(
            forecast_paths=forecasts,
            horizons=horizons,
            subset=subset,
            pnl_type=pnl_type,
        )

        assets = list(pnl_by_asset.keys())
        n_assets = len(assets)
        n_horizons = next(iter(pnl_by_asset.values())).shape[1]
        n_paths = forecasts.n_simulations
        prob = forecasts.path_probs

        inc_returns = np.empty((n_paths, n_horizons, n_assets), dtype=float)
        for col_idx, asset_returns in enumerate(pnl_by_asset.values()):
            inc_returns[:, :, col_idx] = asset_returns

        if expected_returns == "forecast":
            mean = np.empty((n_horizons, n_assets), dtype=float)
            for h in range(n_horizons):
                mean[h] = weighted_mean(data=inc_returns[:, h, :], prob=prob)
        else:
            if history is None:
                raise ValueError(
                    "history is required when expected_returns='historical'."
                )
            mean = cls._historical_mean(
                history,
                assets=assets,
                horizons=n_horizons,
                mean_decay=mean_decay,
            )

        covariances = correlations = cov_factor = None
        if risk in {"covariance", "both"}:
            covariances = np.empty((n_horizons, n_assets, n_assets), dtype=float)
            correlations = np.empty((n_horizons, n_assets, n_assets), dtype=float)
            for h in range(n_horizons):
                pnl_matrix = inc_returns[:, h, :]
                covariances[h] = weighted_covariance(data=pnl_matrix, prob=prob)
                correlations[h] = weighted_correlation(data=pnl_matrix, prob=prob)
            cov_factor = np.stack(
                [psd_sqrt_factor(covariances[h]) for h in range(n_horizons)]
            )

        scenario_returns = inc_returns if risk in {"cvar", "both"} else None
        scenario_probs = prob if scenario_returns is not None else None

        (
            assets,
            mean,
            covariances,
            correlations,
            scenario_returns,
            cov_factor,
        ) = cls._filter_policy_fields_by_expectation_tolerance(
            assets=assets,
            mean=mean,
            covariances=covariances,
            correlations=correlations,
            scenario_returns=scenario_returns,
            cov_factor=cov_factor,
            expectation_tolerance=expectation_tolerance,
        )

        return cls.from_arrays(
            assets=assets,
            mean=mean,
            covariances=covariances,
            correlations=correlations,
            scenario_returns=scenario_returns,
            scenario_probs=scenario_probs,
            cash_return=cls.get_cash_return(
                path=cash_path,
                step_size=step_size,
                periods_per_year=periods_per_year,
                as_of=as_of,
            ),
            cov_factor=cov_factor,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        assets: list[str],
        mean: NDArray[np.floating] | None = None,
        covariances: NDArray[np.floating] | None = None,
        correlations: NDArray[np.floating] | None = None,
        scenario_returns: NDArray[np.floating] | None = None,
        scenario_probs: ProbVector | None = None,
        cash_return: NDArray[np.floating] | float | None = None,
        cov_factor: NDArray[np.floating] | None = None,
    ) -> "PolicyInputs":
        cash_array = (
            None if cash_return is None else np.asarray(cash_return, dtype=float)
        )
        if cov_factor is None and covariances is not None:
            cov_factor = np.stack(
                [psd_sqrt_factor(covariances[h]) for h in range(covariances.shape[0])]
            )
        return cls(
            assets=assets,
            mean=mean,
            covariances=covariances,
            correlations=correlations,
            scenario_returns=scenario_returns,
            scenario_probs=scenario_probs,
            cash_return=cash_array,
            cov_factor=cov_factor,
        )


@dataclass(frozen=True, slots=True)
class PolicyInputConfig:
    cash_path: str
    expected_returns: ExpectedReturnSource = "forecast"
    risk: RiskSource = "both"
    horizons: int | None = None
    subset: AssetSubset = "tradable"
    pnl_type: PnL_OPTIONS = "relative"
    expectation_tolerance: float | None = 1.0
    mean_decay: float = 1.0
    step_size: int = 1
    periods_per_year: int = 252
