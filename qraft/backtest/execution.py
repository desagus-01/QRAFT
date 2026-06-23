from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qraft.construction.optimization.objectives.specs import (
    TransactionCost,
    transaction_cost_value,
)
from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import ForecastPaths


@dataclass(frozen=True, slots=True)
class BacktestPeriod:
    decision_bar: datetime
    execution_bar: datetime
    state_before: PortfolioState
    decision: PolicyDecision
    executed_share_trades: NDArray[np.floating]
    state_after: PortfolioState
    solver_status: str | None
    cost: float = 0.0
    forecasts: ForecastPaths | None = None
    decision_error: str | None = None


BacktestWarning = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    policy_name: str
    asset_order: list[str]
    nav_dates: list[datetime]
    nav: NDArray[np.floating]
    periods: list[BacktestPeriod]
    warnings_log: list[BacktestWarning] = field(default_factory=list)

    @property
    def period_decision_bars(self) -> list[datetime]:
        return [p.decision_bar for p in self.periods]

    @property
    def period_execution_bars(self) -> list[datetime]:
        return [p.execution_bar for p in self.periods]

    @property
    def period_returns(self) -> NDArray[np.floating]:
        """Simple return per holding period between rebalances.

        For the first period the holding period starts from the initial NAV
        recorded at the first trading bar.  For subsequent periods it starts
        from the portfolio value recorded immediately *after* the previous
        period's trade.  Each period ends at the portfolio value recorded
        immediately *before* the current period's trade.
        """
        if not self.periods:
            return np.array([], dtype=float)
        n = len(self.periods)
        start_values = np.empty(n)
        end_values = np.empty(n)
        for i, p in enumerate(self.periods):
            if i == 0:
                start_values[i] = float(self.nav[0])
            else:
                start_values[i] = float(self.periods[i - 1].state_after.portfolio_value)
            end_values[i] = float(p.state_before.portfolio_value)
        return end_values / start_values - 1.0

    @property
    def period_turnovers(self) -> NDArray[np.floating]:
        """One-way turnover fraction at each rebalance.

        Defined as half the total absolute trade value divided by portfolio
        value immediately before the trade — the standard convention in
        portfolio optimisation (e.g. cvxportfolio).
        """
        if not self.periods:
            return np.array([], dtype=float)
        turnovers = np.empty(len(self.periods))
        for i, p in enumerate(self.periods):
            pv = float(p.state_before.portfolio_value)
            trade_value = float(
                np.abs(p.executed_share_trades * p.state_before.initial_prices).sum()
            )
            turnovers[i] = 0.5 * trade_value / pv if pv > 0 else 0.0
        return turnovers

    @property
    def period_costs(self) -> NDArray[np.floating]:
        """Realised trading cost charged at each rebalance, in NAV units.

        Zero when no ``transaction_cost`` is passed to ``run_backtest``."""
        return np.array([p.cost for p in self.periods], dtype=float)

    @property
    def period_target_weights(self) -> NDArray[np.floating]:
        """Target total weights per period, shape (n_periods, n_assets + 1).

        The last column is the target cash weight.
        """
        n = len(self.periods)
        n_assets = len(self.asset_order)
        out = np.empty((n, n_assets + 1))
        for i, p in enumerate(self.periods):
            out[i, :-1] = _target_weights_full(p.decision, self.asset_order)
            out[i, -1] = p.decision.target_cash_weight
        return out

    @property
    def period_weights(self) -> NDArray[np.floating]:
        """Actual total weights *after* each rebalance, shape (n_periods, n_assets + 1).

        The last column is the cash weight.
        """
        n = len(self.periods)
        n_assets = len(self.asset_order)
        out = np.empty((n, n_assets + 1))
        for i, p in enumerate(self.periods):
            out[i, :-1] = p.state_after.asset_weights
            out[i, -1] = float(p.state_after.cash_weight)
        return out

    @property
    def period_cash(self) -> NDArray[np.floating]:
        """Cash balance after each rebalance."""
        return np.array([p.state_after.cash for p in self.periods], dtype=float)


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
    asset_value = shares * prices
    nav = float(asset_value.sum() + cash)
    target_value = _target_weights_full(decision, asset_order) * nav
    trade_value = target_value - asset_value
    executed = trade_value / prices
    return executed, shares + executed, cash - float(trade_value.sum())


def execute_with_costs(
    decision: PolicyDecision,
    shares: NDArray[np.floating],
    cash: float,
    prices: NDArray[np.floating],
    asset_order: list[str],
    transaction_cost: TransactionCost | None,
    *,
    sigma: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float, float]:
    """Trade to target (as ``execute_frictionless``) then debit realised cost.

    Returns ``(executed_shares, new_shares, new_cash, cost)``. Post-trade
    ``NAV = NAV_pre - cost``. With ``transaction_cost is None`` this is exactly
    frictionless. ``sigma`` (forecast per-asset vol, aligned to ``asset_order``)
    powers the market-impact term; ignored when ``market_impact == 0``.
    """
    executed, new_shares, new_cash = execute_frictionless(
        decision, shares, cash, prices, asset_order
    )
    nav = float((shares * prices).sum() + cash)
    if transaction_cost is None or nav <= 0.0:
        return executed, new_shares, new_cash, 0.0
    cost = transaction_cost_value(
        transaction_cost,
        weight_trades=(executed * prices) / nav,
        nav=nav,
        prices=prices,
        sigma=sigma,
    )
    return executed, new_shares, new_cash - cost, cost
