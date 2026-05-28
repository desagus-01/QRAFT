import logging
import warnings
from re import fullmatch

import numpy as np
from numpy._typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
from typing_extensions import Literal

from policy import MeanModelConfig
from time_series.models.fitted_types import AutoARMARes

logger = logging.getLogger(__name__)


def _build_arma_parameters(
    parameter_names: list[str],
    ar_estimates: NDArray[np.floating],
    ma_estimates: NDArray[np.floating],
) -> dict[str, float]:
    params = {}
    for param in parameter_names:
        m = fullmatch(r"(ar|ma)\.L(\d+)", param)
        if m:
            kind, lag = m.group(1), int(m.group(2))
            if lag < 1:
                raise ValueError(f"Invalid lag in {param}")
            if kind == "ar":
                params[param] = ar_estimates[lag - 1]
            if kind == "ma":
                params[param] = ma_estimates[lag - 1]

    return params


def _arma_top_candidates(
    asset_array: NDArray[np.floating],
    max_ar_order: int,
    max_ma_order: int,
    information_criteria: Literal["bic", "aic"] = "bic",
    top_n_models: int = 3,
) -> list[
    tuple[
        int,
        int,
    ]
]:
    """
    Identify the top ARMA (p, q) orders with the lowest information criterion.

    Uses a fast, approximate estimation method to evaluate candidate ARMA
    models and returns the p and q orders of the best-performing models.
    """
    # Get top n model orders with lowest information criteria
    best_order = arma_order_select_ic(
        y=asset_array,
        max_ar=max_ar_order,
        max_ma=max_ma_order,
        ic=information_criteria,
        trend="n",
        model_kw={"enforce_stationarity": False, "enforce_invertibility": False},
        fit_kw={"method": "hannan_rissanen", "low_memory": True},
    )[information_criteria]

    top = best_order.stack().nsmallest(top_n_models)
    return [(int(ar_order), int(ma_order)) for (ar_order, ma_order) in top.index]


def _arma_filter(
    params: dict[str, float],
    cfg: MeanModelConfig,
) -> bool:
    """
    Filter ARMA candidates using root-based stationarity / invertibility checks.
    """
    ar_vals = [float(v) for k, v in sorted(params.items()) if k.startswith("ar.L")]
    ma_vals = [float(v) for k, v in sorted(params.items()) if k.startswith("ma.L")]

    try:
        if ar_vals:
            ar_poly = np.r_[1.0, -np.asarray(ar_vals, dtype=float)]
            ar_roots = np.roots(ar_poly)
            if np.any(np.abs(ar_roots) <= 1.0 + cfg.arma_stationarity_buffer):
                return False

        if ma_vals:
            ma_poly = np.r_[1.0, np.asarray(ma_vals, dtype=float)]
            ma_roots = np.roots(ma_poly)
            if np.any(np.abs(ma_roots) <= 1.0 + cfg.arma_invertibility_buffer):
                return False
    except (ValueError, np.linalg.LinAlgError):
        return False

    return True


def auto_arma(
    asset_array: NDArray[np.floating],
    cfg: MeanModelConfig | None = None,
    max_ar_order: int | None = None,
    max_ma_order: int | None = None,
    top_n_models: int | None = None,
    information_criteria: Literal["bic", "aic"] | None = None,
) -> list[AutoARMARes]:
    """Fit ARMA models for the top candidate orders and collect their results."""
    if cfg is None:
        cfg = MeanModelConfig()
    # Allow legacy callers to override individual fields via keyword args
    if any(
        v is not None
        for v in [max_ar_order, max_ma_order, top_n_models, information_criteria]
    ):
        cfg = MeanModelConfig(
            max_ar_order=max_ar_order if max_ar_order is not None else cfg.max_ar_order,
            max_ma_order=max_ma_order if max_ma_order is not None else cfg.max_ma_order,
            search_n_models=top_n_models
            if top_n_models is not None
            else cfg.search_n_models,
            information_criteria=information_criteria
            if information_criteria is not None
            else cfg.information_criteria,
            arma_stationarity_buffer=cfg.arma_stationarity_buffer,
            arma_invertibility_buffer=cfg.arma_invertibility_buffer,
            ljung_box_lags=cfg.ljung_box_lags,
            min_ljung_box_rejections=cfg.min_ljung_box_rejections,
        )

    candidates_for_arma = _arma_top_candidates(
        asset_array,
        cfg.max_ar_order,
        cfg.max_ma_order,
        cfg.information_criteria,
        cfg.search_n_models,
    )

    arma_res = []

    for ar_order, ma_order in candidates_for_arma:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)

                model = ARIMA(
                    asset_array,
                    order=(ar_order, 0, ma_order),
                    seasonal_order=[0, 0, 0, 0],
                    trend="n",
                )
                res = model.fit(method="statespace")
        except ConvergenceWarning:
            logger.info(
                "Model (%s, %s) failed to converge. Will be dropped from candidates list",
                ar_order,
                ma_order,
            )
            continue
        except Exception as exc:
            logger.info(
                "Model (%s, %s) failed with error=%s. Will be dropped from candidates list",
                ar_order,
                ma_order,
                type(exc).__name__,
            )
            continue

        params = _build_arma_parameters(model.param_names, res.arparams, res.maparams)  # type: ignore[attr-defined]

        if not _arma_filter(params, cfg):
            logger.info(
                "Model (%s, %s) failed ARMA admissibility checks; will drop it",
                ar_order,
                ma_order,
            )
            continue

        arma_res.append(
            AutoARMARes(
                model_order=(ar_order, ma_order),
                params=params,
                degrees_of_freedom=ar_order + ma_order,
                criteria=cfg.information_criteria,
                criteria_res=float(getattr(res, cfg.information_criteria)),
                p_values=res.pvalues,  # type: ignore[attr-defined]
                residuals=res.resid,  # type: ignore[attr-defined]
                residual_scale=float(np.std(res.resid, ddof=1)),  # type: ignore[attr-defined]
            )
        )
    if not arma_res:
        raise ValueError("No ARMA models were fitted, likely due to failed convergence")

    return arma_res
