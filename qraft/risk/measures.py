from __future__ import annotations

from typing import Literal

import numpy as np
from core.probability.prob_vector import ProbVector
from numpy.lib.array_utils import normalize_axis_index
from numpy.typing import NDArray


def tail_cutoff(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    method: Literal["empirical", "quantile"],
    alpha: float,
    axis: int,
    distribution_type: Literal["pnl", "loss"],
) -> NDArray[np.floating]:
    q = alpha if distribution_type == "pnl" else 1 - alpha

    if method == "empirical" and prob is not None:
        return np.quantile(
            distribution,
            q,
            axis=axis,
            method="inverted_cdf",
            weights=prob,
        )

    if method == "quantile" and prob is None:
        return np.quantile(distribution, q, axis=axis)

    raise ValueError("Must choose either empirical with prob or quantile without prob.")


def var(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    method: Literal["empirical", "quantile"] = "quantile",
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    axis = normalize_axis_index(axis, distribution.ndim)

    cutoff = tail_cutoff(
        distribution=distribution,
        prob=prob,
        method=method,
        alpha=alpha,
        axis=axis,
        distribution_type=distribution_type,
    )

    return cutoff if distribution_type == "loss" else -cutoff


def cvar(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    method: Literal["empirical", "quantile"] = "quantile",
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    axis = normalize_axis_index(axis, distribution.ndim)

    cutoff = tail_cutoff(
        distribution=distribution,
        prob=prob if method == "empirical" else None,
        method=method,
        alpha=alpha,
        axis=axis,
        distribution_type=distribution_type,
    )

    expanded_cutoff = np.expand_dims(cutoff, axis=axis)

    if distribution_type == "pnl":
        tail_mask = distribution <= expanded_cutoff
        tail_values = np.where(tail_mask, distribution, np.nan)
        return -np.nanmean(tail_values, axis=axis)

    tail_mask = distribution >= expanded_cutoff
    tail_values = np.where(tail_mask, distribution, np.nan)
    return np.nanmean(tail_values, axis=axis)
