from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.linalg import sqrtm

from scenarios.types import ProbVector

EquationTypes = Literal["nc", "c", "ct", "ctt"]


class LeastSquaresRes(NamedTuple):
    results: NDArray[np.floating]
    residuals: NDArray[np.floating]
    sum_of_squared_residuals: float


class OLSEquation(NamedTuple):
    ind_var: NDArray[np.floating]
    dep_vars: NDArray[np.floating]


@dataclass
class OLSResults:
    feature_names_order: list[str] | None
    res: NDArray[np.floating]
    std_errors: NDArray[np.floating]
    t_stats: NDArray[np.floating]
    p_values: NDArray[np.floating]
    residuals: NDArray[np.floating]
    r_squared: float
    aic: float
    bic: float

    def __post_init__(self) -> None:
        shape = self.res.shape
        if (
            self.std_errors.shape != shape
            or self.t_stats.shape != shape
            or self.p_values.shape != shape
        ):
            raise ValueError(
                f"""
                OLSResults values must all have the same shape:
                res : {shape}
                std_errors: {self.std_errors.shape}
                t_stats: {self.t_stats.shape}
                p_values: {self.p_values.shape}
                """
            )
        if (
            self.feature_names_order is not None
            and len(self.feature_names_order) != self.res.size
        ):
            raise ValueError(
                f"feature_names length {len(self.feature_names_order)} does not match "
                f"number of coefficients {self.res.size}"
            )


class AICBIC(NamedTuple):
    aic: float
    bic: float


def get_aic(log_likelihood: float, n_parameters: int) -> float:
    return -2 * log_likelihood + 2 * n_parameters


def get_bic(log_likelihood: float, n_obs: int, n_parameters: int) -> float:
    return -2 * log_likelihood + np.log(n_obs) * n_parameters


def add_deterministics_to_eq(
    independent_vars: NDArray[np.floating], eq_type: EquationTypes
) -> NDArray[np.floating]:
    n_obs = independent_vars.shape[0]

    time_index = np.arange(1, n_obs + 1, dtype=float).reshape(-1, 1)
    cols = [np.ones((n_obs, 1), dtype=float)]

    if eq_type == "nc":
        return independent_vars
    if eq_type in ("ct", "ctt"):
        cols.append(time_index)
    if eq_type == "ctt":
        cols.append(time_index**2)

    deterministics = np.hstack(cols)
    return np.hstack([deterministics, independent_vars])


def ols_log_likelihood(sum_of_squared_residuals: float, n_obs: int) -> float:
    return (
        -0.5
        * n_obs
        * (np.log(2 * np.pi) + 1.0 + np.log(sum_of_squared_residuals / n_obs))
    )


def get_aic_bic(
    sum_of_squared_residuals: float,
    n_obs: int,
    n_parameters: int,
) -> AICBIC:
    llf = ols_log_likelihood(
        sum_of_squared_residuals=sum_of_squared_residuals,
        n_obs=n_obs,
    )
    aic = get_aic(log_likelihood=llf, n_parameters=n_parameters)
    bic = get_bic(log_likelihood=llf, n_obs=n_obs, n_parameters=n_parameters)
    return AICBIC(aic=aic, bic=bic)


def get_p_values(
    t_stats: NDArray[np.floating],
    n_obs: int,
    n_parameters: int,
) -> NDArray[np.floating]:
    df = n_obs - n_parameters
    return np.asarray(2 * stats.t.sf(np.abs(t_stats), df=df), dtype=float)


def weighted_mean(
    data: NDArray[np.floating],
    prob: ProbVector,
) -> NDArray[np.floating]:
    w = prob.reshape(-1, 1) if data.ndim > 1 else prob
    return (w * data).sum(axis=0)


def weighted_covariance(
    data: NDArray[np.floating],
    prob: ProbVector,
    center: bool = True,
) -> NDArray[np.floating]:
    data = data.reshape(-1, 1) if data.ndim == 1 else data
    w_mean = weighted_mean(data, prob)
    centered = data - w_mean
    data_for_cov = centered if center else data
    w = prob.reshape(-1, 1)

    return data_for_cov.T @ (w * data_for_cov)


def weighted_variance(
    covariance: NDArray[np.floating] | None = None,
    data: NDArray[np.floating] | None = None,
    prob: ProbVector | None = None,
    center: bool = True,
) -> NDArray[np.floating]:
    if covariance is None:
        if data is None or prob is None:
            raise ValueError(
                "Either `covariance` or both `data` and `prob` must be provided"
            )
        covariance = weighted_covariance(data, prob, center)
    return np.diag(covariance)


def weighted_correlation(
    data: NDArray[np.floating],
    prob: ProbVector,
) -> NDArray[np.floating]:
    covariance = weighted_covariance(data, prob, center=True)
    standard_devs = np.sqrt(np.diag(covariance))
    return np.diag(1.0 / standard_devs) @ covariance @ np.diag(1.0 / standard_devs)


class RiccatiResult(NamedTuple):
    root: NDArray[np.floating]
    standard_devs: NDArray[np.floating]


def riccati_root(
    data: NDArray[np.floating],
    prob: ProbVector,
) -> RiccatiResult:
    covariance = weighted_covariance(data, prob, center=True)
    standard_devs = np.sqrt(np.diag(covariance))
    correlation = (
        np.diag(1.0 / standard_devs) @ covariance @ np.diag(1.0 / standard_devs)
    )
    return RiccatiResult(
        root=np.asarray(sqrtm(correlation).real), standard_devs=standard_devs
    )


def get_r_squared(
    dependent_var: NDArray[np.floating],
    residuals: NDArray[np.floating],
    prob: ProbVector | None = None,
) -> float:
    if prob is None:
        ss_res = float((residuals**2).sum())
        ss_tot = float(((dependent_var - dependent_var.mean()) ** 2).sum())
    else:
        y_mean = weighted_mean(dependent_var, prob)
        w = prob.reshape(-1, 1)
        ss_res = float((w * residuals**2).sum())
        ss_tot = float((w * (dependent_var - y_mean) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0


def least_squares_weighted_fit(
    dependent_var: NDArray[np.floating],
    independent_vars: NDArray[np.floating],
    prob: ProbVector | None = None,
) -> LeastSquaresRes:
    if dependent_var.ndim == 1:
        dependent_var = dependent_var.reshape(-1, 1)

    if prob is None:
        res = np.linalg.lstsq(a=independent_vars, b=dependent_var, rcond=None)
        beta = np.asarray(res[0], dtype=float)
        fitted = independent_vars @ beta
        residuals = dependent_var - fitted

        if res[1].size == 0:
            ssr = float((residuals.T @ residuals).item())
        else:
            ssr = float(res[1].item())
    else:
        sqrt_w = np.sqrt(prob).reshape(-1, 1)
        x_w = independent_vars * sqrt_w
        y_w = dependent_var * sqrt_w

        beta = np.linalg.lstsq(a=x_w, b=y_w, rcond=None)[0]
        beta = np.asarray(beta, dtype=float)

        fitted = independent_vars @ beta
        residuals = dependent_var - fitted
        ssr = float((prob.reshape(-1, 1) * (residuals**2)).sum())

    return LeastSquaresRes(
        results=beta,
        residuals=residuals,
        sum_of_squared_residuals=ssr,
    )


def ols_standard_errors(
    independent_vars: NDArray[np.floating],
    sum_of_squared_residuals: float,
    n_obs: int,
    prob: ProbVector | None = None,
) -> NDArray[np.floating]:
    x = np.asarray(independent_vars, dtype=float)
    k = x.shape[1]

    if n_obs <= k:
        raise ValueError(
            f"Need n_obs > k for standard errors, got n_obs={n_obs}, k={k}"
        )

    cov_scaler = sum_of_squared_residuals / (n_obs - k)

    if prob is None:
        xtx_inv = np.linalg.pinv(x.T @ x)
    else:
        xtx_inv = np.linalg.pinv(weighted_covariance(x, prob, center=False))

    scaled_cov_inv = cov_scaler * xtx_inv
    return np.sqrt(np.diag(scaled_cov_inv)).reshape(-1, 1)


def weighted_ols(
    dependent_var: NDArray[np.floating],
    independent_vars: NDArray[np.floating],
    feature_names: list[str] | None = None,
    prob: ProbVector | None = None,
) -> OLSResults:
    if dependent_var.ndim == 1:
        dependent_var = dependent_var.reshape(-1, 1)

    n_obs = dependent_var.shape[0]

    ols_res, residuals, sum_of_squared_residuals = least_squares_weighted_fit(
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        prob=prob,
    )

    standard_errors = ols_standard_errors(
        independent_vars=independent_vars,
        sum_of_squared_residuals=sum_of_squared_residuals,
        n_obs=n_obs,
        prob=prob,
    )

    fit_evals = get_aic_bic(
        sum_of_squared_residuals=sum_of_squared_residuals,
        n_obs=n_obs,
        n_parameters=independent_vars.shape[1],
    )

    t_stats = ols_res / standard_errors
    p_values = get_p_values(
        t_stats=t_stats,
        n_obs=n_obs,
        n_parameters=independent_vars.shape[1],
    )
    r_squared = get_r_squared(
        dependent_var=dependent_var,
        residuals=residuals,
        prob=prob,
    )

    return OLSResults(
        feature_names_order=feature_names,
        res=ols_res,
        std_errors=standard_errors,
        t_stats=t_stats,
        p_values=p_values,
        residuals=residuals,
        r_squared=r_squared,
        aic=fit_evals.aic,
        bic=fit_evals.bic,
    )
