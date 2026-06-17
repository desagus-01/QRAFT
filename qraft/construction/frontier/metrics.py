from __future__ import annotations

import math

import numpy as np

from qraft.construction.optimization.moments import HorizonMoments
from qraft.construction.optimization.optimization import MPOResult
from qraft.risk.measures import cvar


def _cash_return_by_horizon(moments: HorizonMoments) -> np.ndarray:
    cash_return = np.asarray(moments.cash_return, dtype=float).ravel()
    if cash_return.size == 1:
        return np.full(moments.n_horizons, cash_return[0])
    if cash_return.size != moments.n_horizons:
        raise ValueError(
            f"cash_return has {cash_return.size} value(s), expected 1 or "
            f"{moments.n_horizons}."
        )
    return cash_return


def ex_ante_metrics(result: MPOResult, moments: HorizonMoments) -> dict[str, float]:
    """Analytic ex-ante metrics for the optimized multi-horizon plan."""
    if result.planned_weights.shape != moments.mean.shape:
        raise ValueError(
            "planned_weights shape must match moments.mean shape; got "
            f"{result.planned_weights.shape} and {moments.mean.shape}."
        )

    cash_return = _cash_return_by_horizon(moments)
    expected_risky = float(np.sum(result.planned_weights * moments.mean))
    expected_cash = float(np.dot(result.planned_cash, cash_return))

    variance = 0.0
    for h in range(moments.n_horizons):
        weights_h = result.planned_weights[h]
        variance += float(weights_h.T @ moments.covariances[h] @ weights_h)

    return {
        "expected_return": expected_risky + expected_cash,
        "volatility": math.sqrt(max(variance, 0.0)),
    }


def ex_post_terminal_cvar(
    result: MPOResult,
    moments: HorizonMoments,
    alpha: float,
) -> float:
    """
    Forecast-scenario CVaR of buy-and-hold terminal PnL for the first target.

    This is an ex-post tail diagnostic over the forecast paths. It does not
    project dynamic rebalancing through later planned weights.
    """
    cash_return = _cash_return_by_horizon(moments)
    terminal_risky_gross = np.prod(1.0 + moments.scenario_returns, axis=1)
    terminal_cash_gross = float(np.prod(1.0 + cash_return))
    terminal_pnl = (
        terminal_risky_gross @ result.target_weights
        + result.target_cash * terminal_cash_gross
        - 1.0
    )
    value = cvar(
        terminal_pnl,
        prob=moments.scenario_probs,
        alpha=alpha,
        distribution_type="pnl",
    )
    return float(value)


def in_model_cvar(
    result: MPOResult,
    moments: HorizonMoments,
    alpha: float,
) -> float:
    """
    Per-horizon CVaR summed over the plan, evaluated as optimized.
    """
    total = 0.0
    for h in range(moments.n_horizons):
        losses_h = -(moments.scenario_returns[:, h, :] @ result.planned_weights[h])
        total += cvar(
            distribution=losses_h,
            prob=moments.scenario_probs,
            alpha=alpha,
            distribution_type="loss",
        )
    return total
