from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from qraft.backtest.execution import (
    BacktestPeriod,
    BacktestResult,
    execute_frictionless,
)
from qraft.backtest.market import MarketData
from qraft.backtest.schedule import RebalanceSchedule
from qraft.construction.policies import PolicyDecision, PolicyProtocol
from qraft.construction.state import PortfolioState

logger = logging.getLogger(__name__)


def run_backtest(
    market: MarketData,
    policy: PolicyProtocol,
    *,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    initial_cash: float = 100.0,
    step_size: int = 1,
) -> BacktestResult:
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

    prev: datetime | None = None
    for i, bar in enumerate(bars):
        if prev is not None:
            cash *= 1.0 + market.realised_cash_return(prev, bar)
        prices = market.prices_at(bar)

        # fill the decision queued one bar earlier
        if pending is not None and bar in exec_to_decision:
            d_bar, decision, status = pending
            before = PortfolioState(asset_order, prices, shares, cash)
            executed, shares, cash = execute_frictionless(
                decision, shares, cash, prices, asset_order
            )
            periods.append(
                BacktestPeriod(
                    decision_bar=d_bar,
                    execution_bar=bar,
                    state_before=before,
                    decision=decision,
                    executed_share_trades=executed,
                    state_after=PortfolioState(asset_order, prices, shares, cash),
                    solver_status=status,
                )
            )
            pending = None

        # decide from information <= bar; queue for the next bar (after warm-up)
        if bar in decision_bars and i + 1 >= warmup:
            snapshot = market.snapshot_at(
                bar, schedule.execution_bar(bar, bars), step_size=step_size
            )
            state = PortfolioState(asset_order, snapshot.prices_t, shares, cash)
            try:
                decision = policy.decide(snapshot, state)
                status = getattr(decision.diagnostics, "status", "ok")
            except Exception as exc:  # one bad forecast/solve must not abort the run
                logger.warning("decision at %s failed (%s); holding.", bar, exc)
                decision = PolicyDecision(
                    asset_order, state.asset_weights, float(state.cash_weight)
                )
                status = "solver_error"
            pending = (bar, decision, status)

        nav_dates.append(bar)
        nav.append(float(shares @ prices + cash))
        prev = bar

    if warmup and not periods:
        logger.warning(
            "No decisions: history never reached policy.min_history=%d.", warmup
        )

    return BacktestResult(
        policy_name=policy.name,
        asset_order=asset_order,
        nav_dates=nav_dates,
        nav=np.array(nav),
        periods=periods,
    )
