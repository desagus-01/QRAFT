from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from qraft.backtest.execution import (
    BacktestPeriod,
    BacktestResult,
    Forecaster,
    advance_holdings,
    execute_frictionless,
)
from qraft.backtest.market import MarketData
from qraft.backtest.schedule import RebalanceSchedule
from qraft.construction.policies import PolicyDecision, PolicyProtocol
from qraft.construction.state import PortfolioState

logger = logging.getLogger(__name__)


def _hold(state: PortfolioState) -> PolicyDecision:
    """Fallback decision: keep current weights (zero turnover)."""
    return PolicyDecision(
        asset_order=state.asset_order,
        target_weights_risk=state.asset_weights,
        target_cash_weight=float(state.cash_weight),
    )


def _decide_or_hold(
    policy: PolicyProtocol, state: PortfolioState, forecasts
) -> tuple[PolicyDecision, str]:
    try:
        decision = policy.decide(state, forecasts)
        return decision, getattr(decision.diagnostics, "status", "ok")
    except Exception as exc:  # a single failed solve must not abort the run
        logger.warning("decide() failed (%s); holding current weights.", exc)
        return _hold(state), "solver_error"


def run_backtest(
    market: MarketData,
    policy: PolicyProtocol,
    *,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    initial_cash: float = 100.0,
    forecaster: Forecaster | None = None,
    step_size: int = 1,
    store_forecasts: bool = True,
) -> BacktestResult:
    needs_forecast = getattr(policy, "requires_forecast", True)
    if needs_forecast and forecaster is None:
        raise ValueError(
            f"policy {policy.name!r} requires a forecaster; pass forecaster=..."
        )
    warmup = getattr(forecaster, "min_history", 0) if needs_forecast else 0

    bars = market.trading_bars
    asset_order = list(market.universe.assets)
    steps = schedule.decision_steps(bars)
    exec_to_decision = {e: d for d, e in steps}
    decision_bars = {d for d, _ in steps}

    shares = np.zeros(len(asset_order))
    cash = initial_cash
    pending: PolicyDecision | None = None
    pending_decision_bar: datetime | None = None
    pending_forecasts = None

    nav_dates: list[datetime] = []
    nav_track: list[float] = []
    periods: list[BacktestPeriod] = []

    prev: datetime | None = None
    for i, bar in enumerate(bars):
        # 1. daily accounting: compound cash + mark (shares unchanged across the bar)
        if prev is None:
            prices = market.prices_at(bar)
        else:
            realised = market.realised_step(prev, bar)
            shares, cash, _ = advance_holdings(shares, cash, realised)
            prices = realised.prices_next

        # 2. one-bar-lag fill of a previously queued decision
        if bar in exec_to_decision and pending is not None:
            state_before = PortfolioState(asset_order, prices, shares, cash)
            ex = execute_frictionless(pending, shares, cash, prices, asset_order)
            shares, cash = ex.shares_after, ex.cash_after
            periods.append(
                BacktestPeriod(
                    decision_bar=pending_decision_bar,
                    execution_bar=bar,
                    state_before=state_before,
                    decision=pending,
                    executed_share_trades=ex.executed_share_trades,
                    state_after=PortfolioState(asset_order, prices, shares, cash),
                    solver_status=getattr(pending.diagnostics, "status", None),
                    forecasts=pending_forecasts if store_forecasts else None,
                )
            )
            pending = None

        # 3. decision from <= bar info, queued for the next bar (gated by warm-up)
        if bar in decision_bars and (i + 1) >= warmup:
            e = schedule.execution_bar(bar, bars)
            snapshot = market.snapshot_at(bar, e, step_size=step_size)
            forecasts = (
                forecaster.forecast(snapshot, market.universe)
                if needs_forecast
                else None
            )
            state = PortfolioState(asset_order, snapshot.prices_t, shares, cash)
            pending, _ = _decide_or_hold(policy, state, forecasts)
            pending_decision_bar, pending_forecasts = bar, forecasts

        nav_dates.append(bar)
        nav_track.append(float(shares @ prices + cash))
        prev = bar

    if needs_forecast and not periods:
        logger.warning(
            "No decisions executed: history never reached forecaster.min_history=%d; "
            "the book stayed in its initial allocation.",
            warmup,
        )

    return BacktestResult(
        policy_name=policy.name,
        asset_order=asset_order,
        nav_dates=nav_dates,
        nav=np.array(nav_track),
        periods=periods,
    )
