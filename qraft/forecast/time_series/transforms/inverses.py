from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

from time_series.transforms.deseason import HarmonicTerm


@dataclass(frozen=True)
class SeasonalInverseSpec:
    terms: list[HarmonicTerm]

    @staticmethod
    def evaluate_seasonal_terms(
        terms: list[HarmonicTerm],
        time: NDArray[np.int_],
    ) -> NDArray[np.floating]:
        time = np.asarray(time).reshape(-1)
        seasonal = np.zeros_like(time, dtype=float)

        for term in terms:
            if term.kind == "cos":
                if term.omega is None:
                    raise ValueError("Cos must have omega")
                seasonal += term.coefficient * np.cos(term.omega * time)
            elif term.kind == "sin":
                if term.omega is None:
                    raise ValueError("Sin must have omega")
                seasonal += term.coefficient * np.sin(term.omega * time)

        return seasonal

    def inverse_for_forecasts(
        self,
        data: NDArray[np.floating],
        n_train: int,
    ) -> NDArray[np.floating]:
        current = np.asarray(data, dtype=float)

        if current.ndim != 2:
            raise ValueError(
                f"SeasonalInverseSpec expects 2D forecasts of shape "
                f"(n_sims, horizon), got shape {current.shape}"
            )

        horizon = current.shape[1]
        future_t = np.arange(n_train, n_train + horizon, dtype=float)
        seasonal_future = self.evaluate_seasonal_terms(
            terms=self.terms,
            time=future_t,
        )

        return current + seasonal_future[None, :]


@dataclass(frozen=True)
class PolynomialInverseSpec:
    order: int
    betas: NDArray[np.floating]

    def inverse_for_forecasts(
        self,
        data: NDArray[np.floating],
        start_x: int,
    ) -> NDArray[np.floating]:
        current = np.asarray(data, dtype=float)

        if current.ndim != 2:
            raise ValueError(
                f"PolynomialInverseSpec expects 2D forecasts of shape "
                f"(n_sims, horizon), got shape {current.shape}"
            )

        polynomials = np.asarray(self.betas, dtype=float).reshape(-1)
        horizon = current.shape[1]
        future_x = np.arange(start_x, start_x + horizon)
        trend = np.polyval(polynomials, future_x)

        return current + trend[None, :]


@dataclass(frozen=True)
class DifferenceInverseSpec:
    order: int
    initial_values: NDArray[np.floating]

    def inverse_for_forecasts(
        self,
        data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        current = np.asarray(data, dtype=float)

        if current.ndim != 2:
            raise ValueError(
                f"DifferenceInverseSpec expects 2D forecasts of shape "
                f"(n_sims, horizon), got shape {current.shape}"
            )

        init = np.asarray(self.initial_values, dtype=float).reshape(-1)

        if init.shape[0] != self.order:
            raise ValueError(
                f"Expected {self.order} initial values for differencing order "
                f"{self.order}, got {init.shape[0]}"
            )

        for anchor in init[::-1]:
            current = np.cumsum(current, axis=1) + anchor

        return current


InverseSpec = PolynomialInverseSpec | DifferenceInverseSpec | SeasonalInverseSpec


def _choose_inverse_application_order(
    inverse_specs: list[InverseSpec],
) -> list[InverseSpec]:
    if len(inverse_specs) > 1:
        return sorted(
            inverse_specs,
            key=lambda t: 0 if isinstance(t, SeasonalInverseSpec) else 1,
        )
    return inverse_specs


def apply_inverse_transforms(
    asset_data_dict: dict[str, NDArray[np.floating]],
    n_original: int,
    inverse_specs: dict[str, list[InverseSpec]],
    back_to_price: bool = True,
) -> dict[str, NDArray[np.floating]]:
    restored_paths: dict[str, NDArray[np.floating]] = {}

    for asset, transforms in inverse_specs.items():
        current = np.asarray(asset_data_dict[asset], dtype=float)
        ordered_transforms = _choose_inverse_application_order(transforms)

        for inverse_spec in ordered_transforms:
            if isinstance(inverse_spec, SeasonalInverseSpec):
                current = inverse_spec.inverse_for_forecasts(current, n_original)
            elif isinstance(inverse_spec, PolynomialInverseSpec):
                current = inverse_spec.inverse_for_forecasts(current, n_original)
            elif isinstance(inverse_spec, DifferenceInverseSpec):
                current = inverse_spec.inverse_for_forecasts(current)

        if back_to_price:
            current = np.exp(current)

        restored_paths[asset] = current

    return restored_paths
