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


def returns_from_nav(nav: NDArray[np.floating]) -> NDArray[np.floating]:
    """Simple per-bar returns from a NAV path."""
    return np.diff(nav) / nav[:-1]


def drawdown_curve(nav: NDArray[np.floating]) -> NDArray[np.floating]:
    peak = np.maximum.accumulate(nav)
    return nav / peak - 1.0


def max_drawdown(nav: NDArray[np.floating]) -> float:
    return float(np.min(drawdown_curve(nav)))


def annualised_return(nav: NDArray[np.floating], periods_per_year: float) -> float:
    total_return = nav[-1] / nav[0]
    n_years = (len(nav) - 1) / periods_per_year
    return float(total_return ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0


def annualised_vol(
    returns: NDArray[np.floating],
    periods_per_year: float,
) -> float:
    return float(np.nanstd(returns, ddof=1) * np.sqrt(periods_per_year))


def sharpe(
    returns: NDArray[np.floating],
    rf_per_period: float,
    periods_per_year: float,
) -> float:
    excess = returns - rf_per_period
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino(
    returns: NDArray[np.floating],
    rf_per_period: float,
    periods_per_year: float,
) -> float:
    excess = returns - rf_per_period
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    if downside_dev == 0:
        return 0.0
    return float(np.mean(excess) / downside_dev * np.sqrt(periods_per_year))


def calmar(nav: NDArray[np.floating], periods_per_year: float) -> float:
    ann_ret = annualised_return(nav, periods_per_year)
    mdd = abs(max_drawdown(nav))
    return ann_ret / mdd if mdd > 0 else 0.0
