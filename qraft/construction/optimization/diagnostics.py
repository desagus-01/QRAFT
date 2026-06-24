from __future__ import annotations

import math

import numpy as np

from qraft.construction.optimization.moments import PolicyInputs
from qraft.construction.optimization.optimization import MPOResult
from qraft.risk.measures import cvar


def _cash_return_by_horizon(moments: PolicyInputs) -> np.ndarray:
    cash_return = np.asarray(moments.require_cash_return(), dtype=float).ravel()
    if cash_return.size == 1:
        return np.full(moments.n_horizons, cash_return[0])
    if cash_return.size != moments.n_horizons:
        raise ValueError(
            f"cash_return has {cash_return.size} value(s), expected 1 or "
            f"{moments.n_horizons}."
        )
    return cash_return


def forecast_plan_metrics(result: MPOResult, moments: PolicyInputs) -> dict[str, float]:
    """Analytic forecast metrics for the optimized multi-horizon plan."""
    mean = moments.require_mean()
    covariances = moments.require_covariances()
    if result.planned_weights.shape != mean.shape:
        raise ValueError(
            "planned_weights shape must match moments.mean shape; got "
            f"{result.planned_weights.shape} and {mean.shape}."
        )

    cash_return = _cash_return_by_horizon(moments)
    expected_risky = float(np.sum(result.planned_weights * mean))
    expected_cash = float(np.dot(result.planned_cash, cash_return))

    variance = 0.0
    for h in range(moments.n_horizons):
        weights_h = result.planned_weights[h]
        variance += float(weights_h.T @ covariances[h] @ weights_h)

    return {
        "expected_return": expected_risky + expected_cash,
        "volatility": math.sqrt(max(variance, 0.0)),
    }


def forecast_terminal_cvar(
    result: MPOResult,
    moments: PolicyInputs,
    alpha: float,
) -> float:
    """Forecast-scenario CVaR of buy-and-hold terminal PnL for the first target."""
    cash_return = _cash_return_by_horizon(moments)
    scenario_returns, scenario_probs = moments.require_scenarios()
    terminal_risky_gross = np.prod(1.0 + scenario_returns, axis=1)
    terminal_cash_gross = float(np.prod(1.0 + cash_return))
    terminal_pnl = (
        terminal_risky_gross @ result.target_weights
        + result.target_cash * terminal_cash_gross
        - 1.0
    )
    value = cvar(
        terminal_pnl,
        prob=scenario_probs,
        alpha=alpha,
        distribution_type="pnl",
    )
    return float(value)


def in_model_cvar(
    result: MPOResult,
    moments: PolicyInputs,
    alpha: float,
) -> float:
    """Per-horizon CVaR summed over the optimized plan."""
    scenario_returns, scenario_probs = moments.require_scenarios()
    total = 0.0
    for h in range(moments.n_horizons):
        losses_h = -(scenario_returns[:, h, :] @ result.planned_weights[h])
        total += cvar(
            distribution=losses_h,
            prob=scenario_probs,
            alpha=alpha,
            distribution_type="loss",
        )
    return float(total)
