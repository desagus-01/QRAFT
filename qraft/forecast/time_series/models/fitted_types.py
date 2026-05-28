from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy._typing import NDArray

from time_series.models.model_quality import ModelQuality

MeanKind = Literal["none", "demean", "arma"]
VolKind = Literal["none", "garch"]
GARCH_DISTRIBUTIONS = Literal["t", "normal"]


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


MeanModelRes = AutoARMARes | DemeanRes


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
            return 1.0 if (not np.isfinite(scale) or scale <= 0) else scale

        diff = np.diff(non_null_values)
        if diff.size == 0:
            return 1.0

        scale = float(np.std(diff, ddof=1))
        return 1.0 if (not np.isfinite(scale) or scale <= 0) else scale

    def invariant(self, non_null_values: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return standardized innovations for all asset types.

        - GARCH asset: standardized residuals from GARCH fit
        - mean-only asset: residuals / residual_scale
        - no-model asset: [nan] + diff(series) / diff_scale
        """
        if self.volatility_res is not None:
            invariant = np.asarray(self.volatility_res.invariants, dtype=float)

        elif self.mean_res is not None:
            scale = self.innovation_scale(non_null_values)
            invariant = np.asarray(self.mean_res.residuals, dtype=float) / scale

        else:
            diff = np.concatenate(
                [np.array([np.nan], dtype=float), np.diff(non_null_values)]
            )
            scale = self.innovation_scale(non_null_values)
            invariant = diff / scale

        if invariant.shape[0] != non_null_values.shape[0]:
            raise ValueError(
                f"Innovation length mismatch: innov={invariant.shape[0]} "
                f"vs non_null_values={non_null_values.shape[0]}"
            )
        return invariant.astype(float)
