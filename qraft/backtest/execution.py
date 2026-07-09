import numpy as np
from numpy.typing import NDArray

from qraft.construction.policies import PolicyDecision
from qraft.core.weights import target_weights_full


def execute_frictionless(
    decision: PolicyDecision,
    shares: NDArray[np.floating],
    cash: float,
    prices: NDArray[np.floating],
    asset_order: list[str],
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    if not np.all(np.isfinite(prices)):
        raise ValueError("execution prices must contain only finite values")
    if np.any(prices <= 0):
        raise ValueError("execution prices must be strictly positive")
    asset_value = shares * prices
    nav = float(asset_value.sum() + cash)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("NAV before execution must be finite and strictly positive")
    if decision.hold:
        return np.zeros_like(shares), shares.copy(), float(cash)
    target_value = target_weights_full(decision, asset_order) * nav
    trade_value = target_value - asset_value
    executed = trade_value / prices
    return executed, shares + executed, cash - float(trade_value.sum())
