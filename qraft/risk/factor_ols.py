import logging
from dataclasses import dataclass

import numpy as np
from qraft.core.estimation import (
    EquationTypes,
    OLSEquation,
    OLSResults,
    add_deterministics_to_eq,
    weighted_ols,
)
from qraft.core.probability.prob_vector import ProbVector
from numpy.typing import NDArray
from polars import DataFrame
from qraft.risk.feature_selection import (
    Criterion,
    ForwardRegressionResult,
    forward_regression,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FactorAttributionModel:
    exposures: dict[str, float]
    shift_term: float
    residuals: NDArray[np.floating]
    r2: float


@dataclass(frozen=True, slots=True)
class FactorOLSResult:
    ols: OLSResults
    selected_factors: list[str]
    selection_result: ForwardRegressionResult | None


def _get_t0_factor_values(
    original_data: DataFrame, factors_names: list[str], is_log_price: bool = True
) -> dict[str, float]:
    if is_log_price:
        return {
            col: float(np.exp(original_data.select(col).drop_nulls()[-1, 0]))
            for col in factors_names
        }
    else:
        return {
            col: float(original_data.select(col).drop_nulls()[-1, 0])
            for col in factors_names
        }


def factor_cumulative_returns(
    factors_forecast: dict[str, NDArray[np.floating]],
    original_data: DataFrame,
    factors_names: list[str],
    end_horizon: int,
    is_log_price: bool = True,
) -> dict[str, NDArray]:
    if end_horizon <= 0:
        raise ValueError("end_horizon must be a positive integer")

    factors_t0 = _get_t0_factor_values(
        original_data=original_data,
        factors_names=factors_names,
        is_log_price=is_log_price,
    )

    factors_forecast_w_t0 = {}
    idx = end_horizon - 1

    for factor in factors_names:
        forecast = factors_forecast[factor]
        t0_price = factors_t0[factor]
        factors_forecast_w_t0[factor] = (forecast[:, idx] / t0_price) - 1.0

    return factors_forecast_w_t0


def _build_factor_ols_equation(
    factors_cum_forecast: dict[str, NDArray],
    factor_names: list[str],
    portfolio_cum_forecast: NDArray[np.floating],
    eq_type: EquationTypes = "c",
) -> OLSEquation:
    independent_vars = np.column_stack(
        [factors_cum_forecast[name] for name in factor_names]
    )
    dependent_var = portfolio_cum_forecast.reshape(-1, 1)
    if eq_type != "nc":
        independent_vars = add_deterministics_to_eq(
            independent_vars=independent_vars, eq_type=eq_type
        )
    return OLSEquation(ind_var=independent_vars, dep_vars=dependent_var)


def _deterministic_names(eq_type: EquationTypes) -> list[str]:
    if eq_type == "nc":
        return []
    names = ["const"]
    if eq_type in ("ct", "ctt"):
        names.append("trend")
    if eq_type == "ctt":
        names.append("trend_sq")
    return names


def factor_ols_regression(
    factors_cum_forecast: dict[str, NDArray[np.floating]],
    portfolio_cum_forecast: NDArray[np.floating],
    factor_names: list[str],
    auto_select_factors: bool = False,
    criterion: Criterion | None = None,
    prob: ProbVector | None = None,
    eq_type: EquationTypes = "c",
) -> FactorOLSResult:
    if (auto_select_factors) and criterion is None:
        raise ValueError(
            "You must select a criterion if you wish for auto factor selection."
        )

    ols_eq = _build_factor_ols_equation(
        factors_cum_forecast=factors_cum_forecast,
        factor_names=factor_names,
        portfolio_cum_forecast=portfolio_cum_forecast,
        eq_type=eq_type,
    )

    det_names = _deterministic_names(eq_type)
    full_names = det_names + factor_names

    if auto_select_factors and criterion is not None:
        fwd_result = forward_regression(
            dependent_var=ols_eq.dep_vars,
            independent_vars=ols_eq.ind_var,
            feature_names=full_names,
            criterion=criterion,
            prob=prob,
        )
        selected = [n for n in fwd_result.selected_features if n not in det_names]
        dropped = [n for n in factor_names if n not in selected]
        logger.info(
            "Auto factor selection: criterion=%s selected=%s dropped=%s r2=%.4f",
            criterion,
            selected,
            dropped,
            fwd_result.final_model.r_squared,
        )
        return FactorOLSResult(
            ols=fwd_result.final_model,
            selected_factors=selected,
            selection_result=fwd_result,
        )

    ols_result = weighted_ols(
        dependent_var=ols_eq.dep_vars,
        independent_vars=ols_eq.ind_var,
        feature_names=full_names,
        prob=prob,
    )
    logger.info(
        "Factor OLS regression: n_factors=%d r2=%.4f",
        len(factor_names),
        ols_result.r_squared,
    )
    return FactorOLSResult(
        ols=ols_result,
        selected_factors=factor_names,
        selection_result=None,
    )


def extract_factor_attribution_model(
    ols_results: OLSResults,
    selected_factors: list[str],
) -> FactorAttributionModel:
    if ols_results.feature_names_order is None:
        raise ValueError("OLSResults.feature_names_order is required")

    coeffs = ols_results.res.flatten()
    name_to_coeff = {
        name: float(coeffs[i]) for i, name in enumerate(ols_results.feature_names_order)
    }

    return FactorAttributionModel(
        exposures={factor: name_to_coeff[factor] for factor in selected_factors},
        shift_term=name_to_coeff.get("const", 0.0),
        residuals=ols_results.residuals.flatten(),
        r2=ols_results.r_squared,
    )
