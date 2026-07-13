"""Portfolio and tail-risk metric functions."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.lib.array_utils import normalize_axis_index
from numpy.typing import NDArray

from qraft.core.probability.prob_vector import ProbVector

logger = logging.getLogger(__name__)


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
    """Return loss VaR, converting PnL inputs to positive losses when needed."""
    axis = normalize_axis_index(axis, distribution.ndim)
    if distribution.shape[axis] == 0:
        out_shape = distribution.shape[:axis] + distribution.shape[axis + 1 :]
        return np.full(out_shape, np.nan, dtype=float)

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
    """Rockafellar-Uryasev CVaR of losses.

    For losses ``L`` this reports ``zeta + alpha^-1 E[(L - zeta)+]`` where
    ``zeta`` is the loss VaR. PnL inputs are converted to losses via ``-PnL``.
    """
    axis = normalize_axis_index(axis, distribution.ndim)
    if distribution.shape[axis] == 0:
        out_shape = distribution.shape[:axis] + distribution.shape[axis + 1 :]
        return np.full(out_shape, np.nan, dtype=float)

    losses = -distribution if distribution_type == "pnl" else distribution
    zeta = tail_cutoff(
        distribution=losses,
        prob=prob,
        alpha=alpha,
        axis=axis,
        distribution_type="loss",
    )
    excess = np.maximum(losses - np.expand_dims(zeta, axis=axis), 0.0)

    if prob is not None:
        shape = [1] * distribution.ndim
        shape[axis] = prob.shape[0]
        prob_reshaped = prob.reshape(shape)
        expected_excess = np.sum(prob_reshaped * excess, axis=axis)
    else:
        expected_excess = np.mean(excess, axis=axis)

    return zeta + expected_excess / alpha


def returns_from_nav(nav: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return simple per-bar returns from a NAV path."""
    _validate_nav(nav)
    return np.diff(nav) / nav[:-1]


def drawdown_curve(nav: NDArray[np.floating]) -> NDArray[np.floating]:
    _validate_nav(nav)
    peak = np.maximum.accumulate(nav)
    return nav / peak - 1.0


def max_drawdown(nav: NDArray[np.floating]) -> float:
    return float(np.min(drawdown_curve(nav)))


def annualised_return(nav: NDArray[np.floating], periods_per_year: float) -> float:
    _validate_nav(nav)
    if len(nav) < 2:
        logger.warning("annualised_return is NaN: NAV has fewer than 2 points.")
        return float("nan")
    total_return = nav[-1] / nav[0]
    n_years = (len(nav) - 1) / periods_per_year
    if n_years <= 0:
        logger.warning(
            "annualised_return is NaN: periods_per_year produced n_years<=0."
        )
        return float("nan")
    return float(total_return ** (1.0 / n_years) - 1.0)


def annualised_vol(
    returns: NDArray[np.floating],
    periods_per_year: float,
) -> float:
    if returns.size < 2:
        return 0.0
    return float(np.nanstd(returns, ddof=1) * np.sqrt(periods_per_year))


def per_period_sharpe(
    returns: NDArray[np.floating],
    rf_per_period: float = 0.0,
) -> float:
    if returns.size < 2:
        logger.warning("sharpe is NaN: fewer than 2 return observations.")
        return float("nan")
    excess = returns - rf_per_period
    std = np.std(excess, ddof=1)
    if std == 0:
        logger.warning("sharpe is NaN: excess returns have zero sample volatility.")
        return float("nan")
    return float(np.mean(excess) / std)


def sharpe(
    returns: NDArray[np.floating],
    rf_per_period: float,
    periods_per_year: float,
) -> float:
    return per_period_sharpe(returns, rf_per_period) * np.sqrt(periods_per_year)


def sortino(
    returns: NDArray[np.floating],
    rf_per_period: float,
    periods_per_year: float,
) -> float:
    if returns.size < 1:
        logger.warning("sortino is NaN: no return observations.")
        return float("nan")
    excess = returns - rf_per_period
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    if downside_dev == 0:
        logger.warning("sortino is NaN: downside deviation is zero.")
        return float("nan")
    return float(np.mean(excess) / downside_dev * np.sqrt(periods_per_year))


def calmar(nav: NDArray[np.floating], periods_per_year: float) -> float:
    ann_ret = annualised_return(nav, periods_per_year)
    mdd = abs(max_drawdown(nav))
    if mdd <= 0:
        logger.warning("calmar is NaN: max drawdown is zero.")
        return float("nan")
    return ann_ret / mdd


def _validate_nav(nav: NDArray[np.floating]) -> None:
    if not np.all(np.isfinite(nav)):
        raise ValueError("NAV must contain only finite values.")
    if np.any(nav <= 0):
        raise ValueError("NAV must be strictly positive.")
