from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.lib.array_utils import normalize_axis_index
from numpy.typing import NDArray

from qraft.core.probability.prob_vector import ProbVector


def tail_cutoff(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    alpha: float,
    axis: int,
    distribution_type: Literal["pnl", "loss"],
) -> NDArray[np.floating]:
    q = alpha if distribution_type == "pnl" else 1 - alpha

    if prob is not None:
        return np.quantile(
            distribution,
            q,
            axis=axis,
            method="inverted_cdf",
            weights=prob,
        )

    return np.quantile(distribution, q, axis=axis)


def var(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    axis = normalize_axis_index(axis, distribution.ndim)

    cutoff = tail_cutoff(
        distribution=distribution,
        prob=prob,
        alpha=alpha,
        axis=axis,
        distribution_type=distribution_type,
    )

    return cutoff if distribution_type == "loss" else -cutoff


def cvar(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    axis = normalize_axis_index(axis, distribution.ndim)

    cutoff = tail_cutoff(
        distribution=distribution,
        prob=prob,
        alpha=alpha,
        axis=axis,
        distribution_type=distribution_type,
    )

    expanded_cutoff = np.expand_dims(cutoff, axis=axis)

    if distribution_type == "pnl":
        tail_mask = distribution <= expanded_cutoff
    else:
        tail_mask = distribution >= expanded_cutoff

    if prob is not None:
        shape = [1] * distribution.ndim
        shape[axis] = prob.shape[0]
        prob_reshaped = prob.reshape(shape)

        weighted_sum = np.sum(prob_reshaped * distribution * tail_mask, axis=axis)
        weight_sum = np.sum(prob_reshaped * tail_mask, axis=axis)

        if np.any(weight_sum == 0):
            warnings.warn(
                "Tail has zero probability mass for some samples; returning NaN.",
                RuntimeWarning,
            )

        result = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
    else:
        tail_values = np.where(tail_mask, distribution, np.nan)

        if np.any(np.all(~tail_mask, axis=axis)):
            warnings.warn(
                "Tail is empty for some samples; returning NaN.",
                RuntimeWarning,
            )

        result = np.nanmean(tail_values, axis=axis)

    return -result if distribution_type == "pnl" else result
