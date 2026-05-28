from __future__ import annotations

import numpy as np
from numpy._typing import NDArray

from time_series.models.model_types import CompiledParams


def as_sims_by_horizon(
    array: NDArray[np.floating],
) -> tuple[NDArray[np.floating], int, int]:
    """
    Normalize x to shape (n_sims, horizon), dtype float.
    Accepts 1D (horizon,) or 2D (n_sims, horizon).
    """
    x = np.asarray(array, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("Expected 1D or 2D array.")
    n_sims, horizon = x.shape
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return x, n_sims, horizon


def broadcast_last_k_lags(
    lags_1d: NDArray[np.floating] | None, n_sims: int, k: int, *, name: str
) -> NDArray[np.floating]:
    """
    Return shape (n_sims, k) containing the last k values from lags_1d,
    broadcasted across simulations.

    If k == 0 => returns empty (n_sims, 0).
    """
    if k <= 0:
        return np.empty((n_sims, 0), dtype=float)
    if lags_1d is None:
        raise ValueError(f"Missing {name} lags: need {k} values.")
    lags_1d = np.asarray(lags_1d, dtype=float).ravel()
    if lags_1d.size < k:
        raise ValueError(f"Insufficient {name} lags: need {k}, have {lags_1d.size}.")
    return np.broadcast_to(lags_1d[-k:], (n_sims, k)).copy()


def lag_matrix(
    ext: NDArray[np.floating], last_col: int, k: int
) -> NDArray[np.floating]:
    """
    Return a (n_sims, k) matrix of lags from a 2D 'extended' buffer.
    Column order: [t-1, t-2, ..., t-k]
    """
    if k <= 0:
        return ext[:, :0]
    return ext[:, last_col - np.arange(k)]


def take_vector(
    params: CompiledParams, attr: str, expected: int
) -> NDArray[np.floating]:
    x = np.asarray(getattr(params, attr), dtype=float).ravel()
    if x.size != expected:
        raise ValueError(f"{attr} size {x.size} != {expected}")
    return x


def take_scalar(params: CompiledParams, attr: str) -> float:
    return float(getattr(params, attr))


def garch_params(
    params: CompiledParams, order: tuple[int, int, int]
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    p, o, q = order
    omega = take_scalar(params, "omega")
    alpha = take_vector(params, "alpha", p)
    gamma = take_vector(params, "gamma", o)
    beta = take_vector(params, "beta", q)
    return omega, alpha, gamma, beta


def mean_params(
    params: CompiledParams, order: tuple[int, int]
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    p, q = order
    mu = take_scalar(params, "mu")
    ar = take_vector(params, "ar", p)
    ma = take_vector(params, "ma", q)
    return mu, ar, ma
