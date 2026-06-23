from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import Any

import numpy as np

from qraft.backtest.costs import CostModel
from qraft.backtest.execution import (
    BacktestPeriod,
    BacktestResult,
    BacktestWarning,
    execute_frictionless,
)
from qraft.backtest.forecasting import BacktestForecaster  # noqa: F401
from qraft.backtest.market import MarketData
from qraft.backtest.schedule import RebalanceSchedule
from qraft.construction.optimization.moments import PolicyInputs
from qraft.construction.policies import PolicyDecision, PolicyProtocol
from qraft.construction.state import PortfolioState
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _make_warning(
    message: str,
    bar: datetime | None = None,
    asset: str | None = None,
    exception: str | None = None,
    details: dict[str, Any] | None = None,
) -> BacktestWarning:
    record: BacktestWarning = {"message": message}
    if bar is not None:
        record["bar"] = bar
    if asset is not None:
        record["asset"] = asset
    if exception is not None:
        record["exception"] = exception
    if details:
        record["details"] = details
    return record


def _aligned_sigma(
    policy_inputs: PolicyInputs | None,
    asset_order: list[str],
) -> NDArray[np.floating] | None:
    """Forecast per-asset sigma aligned to ``asset_order`` for the impact term.

    Uses the horizon-0 covariance diagonal of the decision's PolicyInputs -- the
    same forecast risk the optimizer's TransactionCost penalty uses. Assets
    absent from the policy inputs get sigma=0; returns None when no covariance is
    available (e.g. risk='cvar')."""
    if policy_inputs is None or policy_inputs.covariances is None:
        return None
    diag = np.sqrt(np.maximum(np.diag(policy_inputs.covariances[0]), 0.0))
    sig = dict(zip(policy_inputs.assets, diag))
    return np.array([sig.get(a, 0.0) for a in asset_order], dtype=float)


_FLOAT_TOL: float = 1e-9


def _charge(cash: float, amount: float, nav: float, label: str, bar: datetime) -> float:
    """Debit a realised cost from cash; absorb float-noise, warn on a real overdraft."""
    cash -= amount
    if cash >= 0.0 or cash > -_FLOAT_TOL * max(abs(nav), 1.0):
        return max(cash, 0.0)
    logger.warning(
        "%s at %s drove cash negative (%.6g); add a MinCashWeight buffer "
        "or allow_borrow.",
        label,
        bar,
        cash,
    )
    return cash


def run_backtest(
    market: MarketData,
    policy: PolicyProtocol,
    *,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    forecaster: BacktestForecaster | None = None,
    initial_cash: float = 100.0,
    step_size: int = 1,
    costs: CostModel | None = None,
) -> BacktestResult:
    if costs is None:
        costs = CostModel.from_policy(policy)

    warmup = policy.min_history

    bars = market.trading_bars
    asset_order = list(market.universe.assets)
    steps = schedule.decision_steps(bars)
    exec_to_decision = {e: d for d, e in steps}
    decision_bars = {d for d, _ in steps}

    shares = np.zeros(len(asset_order))
    cash = initial_cash
    pending = None  # (decision_bar, decision, status)
    periods: list[BacktestPeriod] = []
    nav_dates: list[datetime] = []
    nav: list[float] = []
    holding_costs: list[float] = []
    decision_index = 0
    warnings_log: list[BacktestWarning] = []

    # A bundled policy (e.g. ForecastingMPOPolicy) can produce its own inputs;
    # otherwise inputs come from an explicit forecaster, else there are none.
    inputs_source = (
        forecaster
        if forecaster is not None
        else (policy if hasattr(policy, "policy_inputs_at") else None)
    )

    prev: datetime | None = None
    for i, bar in enumerate(bars):
        if prev is not None:
            cash *= 1.0 + market.realised_cash_return(prev, bar)
        prices = market.prices_at(bar)
        nav_now = float(shares @ prices + cash)

        # Holding cost accrues per bar on the book held over (prev, bar].
        holding = 0.0
        if prev is not None and nav_now > 0.0:
            holding = costs.holding_charge((shares * prices) / nav_now, nav_now)
            cash = _charge(cash, holding, nav_now, "holding cost", bar)
        holding_costs.append(holding)

        # fill the decision queued one bar earlier
        if pending is not None and bar in exec_to_decision:
            d_bar, decision, status, error_msg, sigma = pending
            before = PortfolioState(asset_order, prices, shares, cash)
            nav_pre = float(before.portfolio_value)
            executed, shares, cash = execute_frictionless(
                decision, shares, cash, prices, asset_order
            )
            trade_cost = costs.trade_charge(executed, prices, nav_pre, sigma=sigma)
            cash = _charge(cash, trade_cost, nav_pre, "trade cost", bar)
            periods.append(
                BacktestPeriod(
                    decision_bar=d_bar,
                    execution_bar=bar,
                    state_before=before,
                    decision=decision,
                    executed_share_trades=executed,
                    state_after=PortfolioState(asset_order, prices, shares, cash),
                    solver_status=status,
                    cost=trade_cost,
                    decision_error=error_msg,
                )
            )
            pending = None

        # decide from information <= bar; queue for the next bar (after warm-up)
        if bar in decision_bars and i + 1 >= warmup:
            snapshot = market.snapshot_at(
                bar, schedule.execution_bar(bar, bars), step_size=step_size
            )
            state = PortfolioState(asset_order, snapshot.prices_t, shares, cash)
            error_msg: str | None = None
            sigma: NDArray[np.floating] | None = None
            try:
                policy_inputs = (
                    inputs_source.policy_inputs_at(snapshot, decision_index)
                    if inputs_source is not None
                    else None
                )
                decision = policy.decide(state, policy_inputs)
                status = getattr(decision.diagnostics, "status", "ok")
                sigma = _aligned_sigma(policy_inputs, asset_order)
            except (RuntimeError, ValueError) as exc:
                logger.warning("decision at %s failed (%s); holding.", bar, exc)
                logger.debug(
                    "Traceback for failed decision at %s:\n%s",
                    bar,
                    traceback.format_exc(),
                )
                exc_str = f"{type(exc).__name__}: {exc}"
                warnings_log.append(
                    _make_warning(
                        message="decision failed; holding current weights",
                        bar=bar,
                        exception=exc_str,
                    )
                )
                decision = PolicyDecision(
                    asset_order, state.asset_weights, float(state.cash_weight)
                )
                status = "solver_error"
                error_msg = exc_str
            pending = (bar, decision, status, error_msg, sigma)
            decision_index += 1

        nav_dates.append(bar)
        nav.append(float(shares @ prices + cash))
        prev = bar

    if warmup and not periods:
        msg = f"No decisions: history never reached policy.min_history={warmup}."
        logger.warning(msg)
        warnings_log.append(_make_warning(message=msg))

    return BacktestResult(
        policy_name=policy.name,
        asset_order=asset_order,
        nav_dates=nav_dates,
        nav=np.array(nav),
        periods=periods,
        warnings_log=warnings_log,
        holding_costs=np.array(holding_costs, dtype=float),
    )
