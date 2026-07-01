from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
    dropped_assets: tuple[str, ...] = ()
    asset_diagnostics: tuple[Any, ...] = ()


BacktestWarning = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    policy_name: str
    asset_order: list[str]
    nav_dates: list[datetime]
    nav: NDArray[np.floating]
    periods: list[BacktestPeriod]
    warnings_log: list[BacktestWarning] = field(default_factory=list)
    holding_costs: NDArray[np.floating] = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    periods_per_year: float | None = None

    def window(self, start: datetime, end: datetime) -> "BacktestResult":
        """Return a sub-result over the inclusive date range ``[start, end]``."""
        keep = [start <= d <= end for d in self.nav_dates]
        mask = np.asarray(keep, dtype=bool)
        nav_dates = [d for d, k in zip(self.nav_dates, keep) if k]
        nav = self.nav[mask] if self.nav.size == mask.size else self.nav[:0]
        holding = (
            self.holding_costs[mask]
            if self.holding_costs.size == mask.size
            else self.holding_costs[:0]
        )
        periods = [p for p in self.periods if start <= p.execution_bar <= end]
        warnings = [
            w
            for w in self.warnings_log
            if isinstance(w.get("bar"), datetime) and start <= w["bar"] <= end
        ]
        return BacktestResult(
            policy_name=self.policy_name,
            asset_order=self.asset_order,
            nav_dates=nav_dates,
            nav=nav,
            periods=periods,
            warnings_log=warnings,
            holding_costs=holding,
            periods_per_year=self.periods_per_year,
        )

    @property
    def dropped_assets(self) -> tuple[str, ...]:
        return tuple(
            sorted({asset for p in self.periods for asset in p.dropped_assets})
        )

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
        """Realised **transaction** cost per rebalance (NAV units)."""
        return np.array([p.cost for p in self.periods], dtype=float)

    @property
    def total_transaction_cost(self) -> float:
        return float(self.period_costs.sum())

    @property
    def total_holding_cost(self) -> float:
        return float(self.holding_costs.sum())

    @property
    def total_cost(self) -> float:
        return self.total_transaction_cost + self.total_holding_cost

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
    if not np.all(np.isfinite(prices)):
        raise ValueError("execution prices must contain only finite values")
    if np.any(prices <= 0):
        raise ValueError("execution prices must be strictly positive")
    asset_value = shares * prices
    nav = float(asset_value.sum() + cash)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("NAV before execution must be finite and strictly positive")
    target_value = _target_weights_full(decision, asset_order) * nav
    trade_value = target_value - asset_value
    executed = trade_value / prices
    return executed, shares + executed, cash - float(trade_value.sum())
