from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy._typing import NDArray
from polars.dataframe.frame import DataFrame

from time_series.estimation import (
    OLSEquation,
    OLSResults,
    add_deterministics_to_eq,
    weighted_ols,
)


@dataclass(frozen=True)
class HarmonicTerm:
    kind: Literal["const", "cos", "sin"]
    coefficient: float
    omega: float | None = None


@dataclass(frozen=True)
class DeterministicSeasonalAdjustmentResult:
    residuals: pl.DataFrame
    terms: list[HarmonicTerm]


def build_harmonic_terms(
    frequency_radians: list[float],
    coefficients: NDArray[np.floating],
) -> list[HarmonicTerm]:
    """
    Construct a list of HarmonicTerm objects from frequencies and coefficient vector.

    Parameters
    ----------
    frequency_radians : list[float]
        List of angular frequencies (radians) used in the harmonic regression.
    coefficients : NDArray[np.floating]
        Coefficient vector returned by OLS (order: cos terms then sin terms as constructed).

    Returns
    -------
    list[HarmonicTerm]
        List of HarmonicTerm dataclasses describing the deterministic seasonal components.

    Raises
    ------
    ValueError
        If the provided coefficient vector length does not match the expected count.
    """
    terms: list[HarmonicTerm] = []

    coef_idx = 0

    for w in frequency_radians:
        if np.isclose(w, 0.0):
            continue
        terms.append(
            HarmonicTerm(
                kind="cos",
                omega=float(w),
                coefficient=float(coefficients[coef_idx]),
            )
        )
        coef_idx += 1

    for w in frequency_radians:
        if np.isclose(w, 0.0) or np.isclose(w, np.pi):
            continue
        terms.append(
            HarmonicTerm(
                kind="sin",
                omega=float(w),
                coefficient=float(coefficients[coef_idx]),
            )
        )
        coef_idx += 1

    if coef_idx != len(coefficients):
        raise ValueError(
            f"Coefficient count mismatch when building harmonic terms: "
            f"used {coef_idx}, got {len(coefficients)}"
        )

    return terms


def build_harmonic_regression_equation(
    data: DataFrame,
    frequency_radians: list[float],
    asset: str,
) -> OLSEquation:
    dependent_variable = data.select(pl.col(asset)).to_numpy()

    time_index_df = data.select(pl.col(asset)).with_row_index(name="t")

    cos_cols = []
    sin_cols = []

    for i, w in enumerate(frequency_radians):
        if np.isclose(w, 0.0):
            continue

        cos_cols.append((pl.lit(w) * pl.col("t")).cos().alias(f"cos_w_{i}"))

        if not np.isclose(w, np.pi):
            sin_cols.append((pl.lit(w) * pl.col("t")).sin().alias(f"sin_w_{i}"))

    independent_variables = time_index_df.select(cos_cols + sin_cols).to_numpy()
    independent_variables = add_deterministics_to_eq(
        independent_vars=independent_variables,
        eq_type="nc",  # this should be nc (ie no constant) as we remove any trend prior to this IF NEEDED
    )

    return OLSEquation(ind_var=independent_variables, dep_vars=dependent_variable)


def run_harmonic_regression(
    data: DataFrame, asset: str, frequency_radians: list[float]
) -> OLSResults:
    harmonic_equation = build_harmonic_regression_equation(
        data=data, asset=asset, frequency_radians=frequency_radians
    )
    return weighted_ols(
        dependent_var=harmonic_equation.dep_vars,
        independent_vars=harmonic_equation.ind_var,
    )


def deterministic_seasonal_adjustment(
    data: DataFrame, asset: str, frequency_radians: list[float]
) -> DeterministicSeasonalAdjustmentResult:
    asset_df = data.drop_nulls()

    harmonic_ols = run_harmonic_regression(
        data=asset_df,
        asset=asset,
        frequency_radians=frequency_radians,
    )

    residuals = asset_df.select("date").with_columns(
        pl.Series(name=asset, values=harmonic_ols.residuals.ravel())
    )

    terms = build_harmonic_terms(
        frequency_radians=frequency_radians,
        coefficients=harmonic_ols.res.ravel(),
    )

    return DeterministicSeasonalAdjustmentResult(
        residuals=residuals,
        terms=terms,
    )
