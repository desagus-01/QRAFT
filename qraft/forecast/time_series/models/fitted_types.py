from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy._typing import NDArray

from qraft.forecast.time_series.models.model_quality import ModelQuality

logger = logging.getLogger(__name__)


MeanKind = Literal["none", "demean", "arma", "random_walk"]
VolKind = Literal["none", "garch"]
GARCH_DISTRIBUTIONS = Literal["t", "normal", "skewt"]


class AutoGARCHRes(NamedTuple):
    model_order: tuple[int, int, int]
    degrees_of_freedom: int
    criteria: Literal["aic", "bic"]
    criteria_res: float
    params: dict[str, float]
    p_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    conditional_volatility: NDArray[np.floating]
    invariants: NDArray[np.floating]
    kind: Literal["garch"] = "garch"
    persistence: float = 0.0
    admissible: bool = True
    fallback_reason: str | None = None


class AutoARMARes(NamedTuple):
    model_order: tuple[int, int]
    degrees_of_freedom: int
    criteria: Literal["aic", "bic"]
    criteria_res: float
    params: dict[str, float]
    p_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    residual_scale: float
    kind: Literal["arma"] = "arma"


class DemeanRes(NamedTuple):
    model_order: None
    degrees_of_freedom: int
    params: dict[str, float]
    residuals: NDArray[np.floating]
    residual_scale: float
    kind: Literal["demean"] = "demean"


class RandomWalkRes(NamedTuple):
    model_order: None = None
    degrees_of_freedom: int = 0
    params: dict[str, float] = {}
    residuals: NDArray[np.floating] = np.array([])
    residual_scale: float = 1.0
    kind: Literal["random_walk"] = "random_walk"


MeanModelRes = AutoARMARes | DemeanRes | RandomWalkRes


@dataclass
class UnivariateRes:
    mean_res: MeanModelRes | None
    volatility_res: AutoGARCHRes | None
    quality: ModelQuality | None

    def innovation_scale(self, non_null_values: NDArray[np.floating]) -> float:
        """
        Scale used to standardize / unstandardize innovations for non-GARCH assets.
        For GARCH assets, innovations are already standardized dynamically.
        """
        if self.volatility_res is not None:
            return 1.0

        if self.mean_res is not None:
            scale = float(self.mean_res.residual_scale)
            result = 1.0 if (not np.isfinite(scale) or scale <= 0) else scale
            if result <= 0 or not np.isfinite(result):
                logger.warning("Mean residual_scale=%s clamped to 1.0", scale)
            return result

        raise TypeError("UnivariateRes has no mean or volatility model")

    def invariant(self, non_null_values: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return standardized innovations for all asset types.

        - GARCH asset: standardized residuals from GARCH fit
        - mean-only asset (ARMA / demean): residuals / residual_scale
        - random walk asset: [nan] + diff(series) / diff_scale  (NaN at position 0
          aligns length N with the date grid; downstream NaN-dropping handles it)
        """
        if self.volatility_res is not None:
            invariant = np.asarray(self.volatility_res.invariants, dtype=float)
            n_nan = int(np.isnan(invariant).sum())
            if n_nan:
                logger.warning("GARCH invariants already contain %d NaN(s)", n_nan)

        elif isinstance(self.mean_res, RandomWalkRes):
            diff = np.diff(non_null_values)
            scale = self.innovation_scale(non_null_values)
            invariant = np.concatenate([[np.nan], diff / scale])

        elif self.mean_res is not None:
            scale = self.innovation_scale(non_null_values)
            residuals = np.asarray(self.mean_res.residuals, dtype=float)
            r_nan = int(np.isnan(residuals).sum())
            r_inf = int(np.isinf(residuals).sum())
            if r_nan or r_inf:
                logger.warning(
                    "Mean residuals have nan=%d, inf=%d, scale=%s", r_nan, r_inf, scale
                )
            invariant = residuals / scale
            if np.isnan(scale) or scale == 0:
                logger.error(
                    "Scale is %s — division produces all NaNs for this asset", scale
                )

        else:
            raise TypeError("UnivariateRes has no mean or volatility model")

        if invariant.shape[0] != non_null_values.shape[0]:
            raise ValueError(
                f"Innovation length mismatch: innov={invariant.shape[0]} "
                f"vs non_null_values={non_null_values.shape[0]}"
            )
        return invariant.astype(float)
