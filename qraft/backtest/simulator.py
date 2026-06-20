from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from qraft.backtest.execution import (
    BacktestPeriod,
    BacktestResult,
    Forecaster,
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
    pending = None  # (decision_bar, decision, forecasts, status)
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
            d_bar, decision, fc, status = pending
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
                    forecasts=fc if store_forecasts else None,
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
                forecasts = (
                    forecaster.forecast(snapshot, market.universe)
                    if needs_forecast
                    else None
                )
                decision = policy.decide(state, forecasts)
                status = getattr(decision.diagnostics, "status", "ok")
            except Exception as exc:  # one bad forecast/solve must not abort the run
                logger.warning("decision at %s failed (%s); holding.", bar, exc)
                decision = PolicyDecision(
                    asset_order, state.asset_weights, float(state.cash_weight)
                )
                forecasts, status = None, "solver_error"
            pending = (bar, decision, forecasts, status)

        nav_dates.append(bar)
        nav.append(float(shares @ prices + cash))
        prev = bar

    if needs_forecast and not periods:
        logger.warning(
            "No decisions: history never reached forecaster.min_history=%d.", warmup
        )

    return BacktestResult(
        policy_name=policy.name,
        asset_order=asset_order,
        nav_dates=nav_dates,
        nav=np.array(nav),
        periods=periods,
    )
