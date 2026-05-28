import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve, sqrtm

from scenarios.types import ProbVector
from time_series.estimation import riccati_root


def _validate_torsion_args(
    method: Literal["approximate", "exact"],
    max_iter: int | None,
) -> int | None:
    if (method == "approximate") and (max_iter is not None):
        raise ValueError("Method selected must be `exact` if selecting a max_iter.")
    if (method == "exact") and (max_iter is None):
        return 10_000
    return max_iter


def _torsion_approximate(
    riccati_root_array: NDArray, diag_sigma: NDArray, diag_sigma_inv: NDArray
) -> NDArray[np.floating]:
    return diag_sigma @ np.linalg.inv(riccati_root_array) @ diag_sigma_inv


def _torsion_exact(
    riccati_root_array: NDArray,
    diag_sigma: NDArray,
    diag_sigma_inv: NDArray,
    n: int,
    max_iter: int,
) -> NDArray[np.floating]:
    scaling_vector = np.ones(n)
    f = np.zeros(max_iter)

    for i in range(max_iter):
        scaled_corr = (
            np.diag(scaling_vector)
            @ riccati_root_array
            @ riccati_root_array
            @ np.diag(scaling_vector)
        )
        normalised_scaled_corr = sqrtm(scaled_corr).real
        procrustes_rotation = solve(
            normalised_scaled_corr, np.diag(scaling_vector) @ riccati_root_array
        )

        scaling_vector = np.diag(procrustes_rotation @ riccati_root_array)
        perturbation = np.diag(scaling_vector) @ procrustes_rotation
        f[i] = np.linalg.norm(riccati_root_array - perturbation, "fro")

        if i > 0 and abs(f[i] - f[i - 1]) / f[i] / n <= 1e-8:
            break
        elif (
            i == max_iter - 1
            and abs(f[max_iter - 1] - f[max_iter - 2]) / f[max_iter - 1] / n > 1e-8
        ):
            warnings.warn(f"number of max iterations reached: n_iter = {max_iter}")

    torsion_corr_space = perturbation @ np.linalg.inv(riccati_root_array)
    return diag_sigma @ torsion_corr_space @ diag_sigma_inv


def minimum_torsion_matrix(
    data: NDArray,
    prob: ProbVector,
    method: Literal["approximate", "exact"] = "approximate",
    max_iter: int | None = None,
) -> NDArray[np.floating]:
    max_iter = _validate_torsion_args(method, max_iter)

    riccati = riccati_root(data, prob)
    n = riccati.standard_devs.shape[0]
    diag_sigma = np.diag(riccati.standard_devs)
    diag_sigma_inv = np.diag(1.0 / riccati.standard_devs)
    riccati_root_array = riccati.root

    if method == "approximate":
        return _torsion_approximate(riccati_root_array, diag_sigma, diag_sigma_inv)

    if method == "exact" and max_iter is not None:
        return _torsion_exact(
            riccati_root_array, diag_sigma, diag_sigma_inv, n, max_iter
        )

    raise ValueError(f"Unknown method: {method}")
