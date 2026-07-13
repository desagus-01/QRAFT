from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.costs import CostModel
from qraft.backtest.engine.policy_step import (
    PendingDecision,
    decide_or_hold,
    make_warning,
)
from qraft.backtest.engine.schedule import DecisionPoint, decision_points
from qraft.backtest.execution import execute_frictionless
from qraft.backtest.inputs import OptimizerInputsProvider
from qraft.backtest.result import BacktestPeriod, BacktestResult, BacktestWarning
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.policies import PolicyProtocol
from qraft.construction.state import PortfolioState
from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.utils.log import info_event, warning_event

logger = logging.getLogger(__name__)


def run_backtest(
    market: MarketData,
    policy: PolicyProtocol,
    *,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    inputs: OptimizerInputsProvider | None = None,
    initial_cash: float = 100.0,
    step_size: int = 1,
    costs: CostModel | None = None,
    points: list[DecisionPoint] | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    if config is None:
        config = BacktestConfig(
            schedule=schedule,
            initial_cash=initial_cash,
            periods_per_year=market.config.periods_per_year,
        )
    if costs is None:
        costs = CostModel.from_policy(policy)

    warmup = policy.min_history

    bars = market.trading_bars
    asset_order = list(market.universe.assets)
    if points is None:
        points = decision_points(market, schedule, warmup, step_size=step_size)
    points_by_bar = {point.decision_bar: point for point in points}
    exec_to_decision = {point.execution_bar: point.decision_bar for point in points}

    shares = np.zeros(len(asset_order))
    cash = initial_cash
    pending: PendingDecision | None = None
    periods: list[BacktestPeriod] = []
    nav_dates: list[datetime] = []
    nav: list[float] = []
    holding_costs: list[float] = []

    info_event(
        logger,
        "backtest.started",
        "Backtest started",
        policy=policy.name,
        bars=len(bars),
        assets=len(asset_order),
        decisions=len(points),
        warmup=warmup,
        initial_cash=initial_cash,
    )

    warnings_log: list[BacktestWarning] = []
    invariance_drops: list[object] = []

    prev: datetime | None = None
    for bar in bars:
        prices, nav_now, cash = _accrue_and_price(prev, bar, market, shares, cash)

        holding, cash = _apply_holding_cost(
            prev,
            bar,
            costs,
            shares,
            prices,
            nav_now,
            cash,
            market.config.cash_day_count,
            market.config.periods_per_year,
        )
        holding_costs.append(holding)

        shares, cash, period = _fill_decision(
            pending,
            bar,
            exec_to_decision,
            shares,
            cash,
            prices,
            nav_now,
            asset_order,
            costs,
        )
        if period is not None:
            periods.append(period)
            pending = None

        if bar in points_by_bar:
            point = points_by_bar[bar]
            state = PortfolioState(asset_order, point.snapshot.prices_t, shares, cash)
            policy_step = decide_or_hold(policy, inputs, state, point, asset_order)
            if policy_step.optimizer_inputs is not None:
                invariance_drops.extend(
                    getattr(policy_step.optimizer_inputs, "invariance_drops", ())
                )
            if policy_step.warning is not None:
                warnings_log.append(policy_step.warning)
            pending = PendingDecision(
                decision_bar=bar,
                decision=policy_step.decision,
                solver_status=policy_step.solver_status,
                decision_error=policy_step.decision_error,
                sigma=policy_step.sigma,
                view_diagnostics=getattr(
                    policy_step.optimizer_inputs, "view_diagnostics", None
                ),
                optimizer_inputs=_retained_optimizer_inputs(
                    policy_step.optimizer_inputs, config
                ),
            )

        nav_dates.append(bar)
        nav.append(float(shares @ prices + cash))
        prev = bar

    if warmup and not periods:
        msg = f"No decisions: history never reached policy.min_history={warmup}."
        warning_event(logger, "backtest.no_decisions", msg, warmup=warmup)
        warnings_log.append(make_warning(message=msg))

    initial_nav = nav[0] if nav else initial_cash
    final_nav = nav[-1] if nav else initial_cash
    decision_failures = sum(
        1 for period in periods if period.decision_error is not None
    )
    held_decisions = sum(1 for period in periods if period.decision.hold)

    info_event(
        logger,
        "backtest.completed",
        "Backtest completed",
        policy=policy.name,
        bars=len(nav),
        periods=len(periods),
        warnings=len(warnings_log),
        decision_failures=decision_failures,
        held_decisions=held_decisions,
        final_nav=f"{final_nav:.6g}",
        total_return=f"{(final_nav / initial_nav - 1.0):.6g}" if initial_nav else "nan",
    )

    return BacktestResult(
        policy_name=policy.name,
        asset_order=asset_order,
        nav_dates=nav_dates,
        nav=np.array(nav),
        periods=periods,
        warnings_log=warnings_log,
        holding_costs=np.array(holding_costs, dtype=float),
        periods_per_year=market.config.periods_per_year,
        invariance_drops=tuple(invariance_drops),
    )


def _charge(cash: float, amount: float, nav: float, label: str, bar: datetime) -> float:
    """Debit a realised cost from cash; absorb float-noise, fail on a real overdraft."""
    cash -= amount
    if cash >= 0.0 or cash > -1e-9 * max(abs(nav), 1.0):
        return max(cash, 0.0)
    raise ValueError(
        f"{label} at {bar} drove cash negative ({cash:.6g}); reduce costs, "
        "turnover, or use a larger cash target."
    )


def _accrue_and_price(
    prev: datetime | None,
    bar: datetime,
    market: MarketData,
    shares: NDArray,
    cash: float,
) -> tuple[NDArray, float, float]:
    if prev is not None:
        cash *= 1.0 + market.realised_cash_return(prev, bar)
    prices = market.prices_at(bar)
    return prices, float(shares @ prices + cash), cash


def _apply_holding_cost(
    prev: datetime | None,
    bar: datetime,
    costs: CostModel,
    shares: NDArray,
    prices: NDArray,
    nav_now: float,
    cash: float,
    day_count: int,
    periods_per_year: float,
) -> tuple[float, float]:
    holding = 0.0
    if prev is not None and nav_now > 0.0 and costs.holding is not None:
        days = (bar - prev).total_seconds() / 86_400.0
        holding_periods = days * periods_per_year / day_count
        holding = costs.holding_charge(
            (shares * prices) / nav_now,
            nav_now,
            n_periods=holding_periods,
            periods_per_year=periods_per_year,
        )
        cash = _charge(cash, holding, nav_now, "holding cost", bar)
    return holding, cash


def _fill_decision(
    pending: PendingDecision | None,
    bar: datetime,
    exec_to_decision: dict[datetime, datetime],
    shares: NDArray,
    cash: float,
    prices: NDArray,
    nav_pre: float,
    asset_order: list[str],
    costs: CostModel,
) -> tuple[NDArray, float, BacktestPeriod | None]:
    if pending is None or bar not in exec_to_decision:
        return shares, cash, None
    before = PortfolioState(asset_order, prices, shares, cash)
    execution = execute_frictionless(
        pending.decision,
        shares,
        cash,
        prices,
        asset_order,
        costs,
        sigma=pending.sigma,
    )
    executed = execution.executed_shares
    shares = execution.shares
    cash = execution.cash
    trade_cost = execution.trade_cost
    diagnostics = getattr(pending.decision, "diagnostics", None)
    dropped_assets = tuple(
        getattr(drop, "asset", str(drop))
        for drop in getattr(diagnostics, "dropped_assets", ())
    )
    asset_diagnostics = tuple(getattr(diagnostics, "asset_diagnostics", ()))
    return (
        shares,
        cash,
        BacktestPeriod(
            decision_bar=pending.decision_bar,
            execution_bar=bar,
            state_before=before,
            decision=pending.decision,
            executed_share_trades=executed,
            state_after=PortfolioState(asset_order, prices, shares, cash),
            solver_status=pending.solver_status,
            cost=trade_cost,
            decision_error=pending.decision_error,
            dropped_assets=dropped_assets,
            asset_diagnostics=asset_diagnostics,
            view_diagnostics=pending.view_diagnostics,
            optimizer_inputs=pending.optimizer_inputs,
        ),
    )


def _retained_optimizer_inputs(
    optimizer_inputs: OptimizerInputs | None,
    config: BacktestConfig,
) -> OptimizerInputs | None:
    if optimizer_inputs is None or not config.retain_optimizer_inputs:
        return None
    if config.retain_optimizer_scenarios or optimizer_inputs.scenario_returns is None:
        return optimizer_inputs
    if _has_non_scenario_horizon_input(optimizer_inputs):
        return replace(optimizer_inputs, scenario_returns=None, scenario_probs=None)
    return None


def _has_non_scenario_horizon_input(optimizer_inputs: OptimizerInputs) -> bool:
    return any(
        value is not None
        for value in (
            optimizer_inputs.mean,
            optimizer_inputs.covariances,
            optimizer_inputs.correlations,
            optimizer_inputs.cov_factor,
            optimizer_inputs.cash_return,
        )
    )
