import numpy as np
from numpy.typing import NDArray

from qraft.construction.policies import PolicyDecision


def _target_weights_full(
    decision: PolicyDecision, asset_order: list[str]
) -> NDArray[np.floating]:
    idx = {a: i for i, a in enumerate(asset_order)}
    w = np.zeros(len(asset_order))
    for a, wi in zip(decision.asset_order, decision.target_weights_risk):
        w[idx[a]] = wi  # assets absent from the decision stay 0 -> sold to cash
    return w


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
    target_value = _target_weights_full(decision, asset_order) * nav
    trade_value = target_value - asset_value
    executed = trade_value / prices
    return executed, shares + executed, cash - float(trade_value.sum())
