from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from qraft.backtest.costs import CostModel
from qraft.construction.policies import PolicyDecision
from qraft.core.weights import target_weights_full


class ExecutionResult(NamedTuple):
    executed_shares: NDArray[np.floating]
    shares: NDArray[np.floating]
    cash: float
    trade_cost: float


def execute_frictionless(
    decision: PolicyDecision,
    shares: NDArray[np.floating],
    cash: float,
    prices: NDArray[np.floating],
    asset_order: list[str],
    costs: CostModel | None = None,
    *,
    sigma: NDArray[np.floating] | None = None,
) -> ExecutionResult:
    if not np.all(np.isfinite(prices)):
        raise ValueError("execution prices must contain only finite values")
    if np.any(prices <= 0):
        raise ValueError("execution prices must be strictly positive")
    asset_value = shares * prices
    nav = float(asset_value.sum() + cash)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("NAV before execution must be finite and strictly positive")
    if decision.hold:
        return ExecutionResult(np.zeros_like(shares), shares.copy(), float(cash), 0.0)
    target_value = target_weights_full(decision, asset_order) * nav
    trade_value = target_value - asset_value
    executed = trade_value / prices
    trade_cost = 0.0
    if costs is not None:
        trade_cost = costs.trade_charge(executed, prices, nav, sigma=sigma)
    if trade_cost <= 0.0:
        return ExecutionResult(
            executed,
            shares + executed,
            cash - float(trade_value.sum()),
            trade_cost,
        )
    if trade_cost >= nav:
        raise ValueError(
            "trade cost at execution exceeded NAV; reduce transaction costs, "
            "turnover, or use a larger cash target."
        )
    net_nav = nav - trade_cost
    target_value = target_weights_full(decision, asset_order) * net_nav
    trade_value = target_value - asset_value
    executed = trade_value / prices
    post_cash = cash - float(trade_value.sum()) - trade_cost
    if post_cash < 0.0 and post_cash > -1e-9 * max(abs(nav), 1.0):
        post_cash = 0.0
    return ExecutionResult(executed, shares + executed, post_cash, trade_cost)
