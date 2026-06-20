from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

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
from qraft.forecast.forecast_paths import ForecastPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Portfolio:
    shares: NDArray[np.floating]
    cash: float


@dataclass(frozen=True, slots=True)
class _PendingRebalance:
    decision_bar: datetime
    decision: PolicyDecision
    forecasts: ForecastPaths | None = None


def _hold(state: PortfolioState) -> PolicyDecision:
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
    except Exception as exc:
        logger.warning("decide() failed (%s); holding current weights.", exc)
        return _hold(state), "solver_error"


def _account(
    portfolio: _Portfolio,
    market: MarketData,
    prev: datetime | None,
    bar: datetime,
) -> tuple[_Portfolio, NDArray[np.floating]]:
    if prev is None:
        return portfolio, market.prices_at(bar)
    realised = market.realised_step(prev, bar)
    shares, cash, _ = advance_holdings(portfolio.shares, portfolio.cash, realised)
    return _Portfolio(shares=shares, cash=cash), realised.prices_next


def _fill(
    portfolio: _Portfolio,
    pending: _PendingRebalance | None,
    bar: datetime,
    exec_to_decision: dict[datetime, datetime],
    asset_order: list[str],
    prices: NDArray[np.floating],
    periods: list[BacktestPeriod],
    store_forecasts: bool,
) -> tuple[_Portfolio, _PendingRebalance | None]:
    if bar not in exec_to_decision or pending is None:
        return portfolio, pending

    state_before = PortfolioState(asset_order, prices, portfolio.shares, portfolio.cash)
    ex = execute_frictionless(
        pending.decision, portfolio.shares, portfolio.cash, prices, asset_order
    )
    portfolio = _Portfolio(shares=ex.shares_after, cash=ex.cash_after)
    periods.append(
        BacktestPeriod(
            decision_bar=pending.decision_bar,
            execution_bar=bar,
            state_before=state_before,
            decision=pending.decision,
            executed_share_trades=ex.executed_share_trades,
            state_after=PortfolioState(
                asset_order, prices, portfolio.shares, portfolio.cash
            ),
            solver_status=getattr(pending.decision.diagnostics, "status", None),
            forecasts=pending.forecasts if store_forecasts else None,
        )
    )
    return portfolio, None


def _decide(
    portfolio: _Portfolio,
    pending: _PendingRebalance | None,
    bar: datetime,
    bar_idx: int,
    decision_bars: set[datetime],
    warmup: int,
    market: MarketData,
    schedule: RebalanceSchedule,
    step_size: int,
    needs_forecast: bool,
    forecaster: Forecaster | None,
    policy: PolicyProtocol,
    asset_order: list[str],
    bars: list[datetime],
) -> _PendingRebalance | None:
    if bar not in decision_bars or (bar_idx + 1) < warmup:
        return pending

    exec_bar = schedule.execution_bar(bar, bars)
    snapshot = market.snapshot_at(bar, exec_bar, step_size=step_size)
    forecasts = (
        forecaster.forecast(snapshot, market.universe) if needs_forecast else None
    )
    state = PortfolioState(
        asset_order, snapshot.prices_t, portfolio.shares, portfolio.cash
    )
    decision, _ = _decide_or_hold(policy, state, forecasts)
    return _PendingRebalance(decision_bar=bar, decision=decision, forecasts=forecasts)


def _record_nav(
    bar: datetime,
    portfolio: _Portfolio,
    prices: NDArray[np.floating],
    nav_dates: list[datetime],
    nav_track: list[float],
) -> None:
    nav_dates.append(bar)
    nav_track.append(float(portfolio.shares @ prices + portfolio.cash))


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

    portfolio = _Portfolio(shares=np.zeros(len(asset_order)), cash=initial_cash)
    pending: _PendingRebalance | None = None
    periods: list[BacktestPeriod] = []
    nav_dates: list[datetime] = []
    nav_track: list[float] = []
    prev: datetime | None = None

    for i, bar in enumerate(bars):
        portfolio, prices = _account(portfolio, market, prev, bar)
        portfolio, pending = _fill(
            portfolio,
            pending,
            bar,
            exec_to_decision,
            asset_order,
            prices,
            periods,
            store_forecasts,
        )
        pending = _decide(
            portfolio,
            pending,
            bar,
            i,
            decision_bars,
            warmup,
            market,
            schedule,
            step_size,
            needs_forecast,
            forecaster,
            policy,
            asset_order,
            bars,
        )
        _record_nav(bar, portfolio, prices, nav_dates, nav_track)
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
