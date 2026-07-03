from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from qraft.backtest.costs import CostModel
from qraft.backtest.execution import (
    BacktestPeriod,
    BacktestResult,
    BacktestWarning,
    execute_frictionless,
)
from qraft.backtest.inputs import PolicyInputsProvider
from qraft.core.market import MarketData
from qraft.construction.inputs import PolicyInputRequirements, build_policy_input_table
from qraft.construction.optimization.inputs import InputPlan, PolicyInputs
from qraft.construction.optimization.optimization import OptimizationFailure
from qraft.construction.policies import PolicyDecision, PolicyProtocol
from qraft.construction.state import PortfolioState
from qraft.core.schedule import RebalanceSchedule
from qraft.core.snapshot import MarketSnapshot
from qraft.forecast.forecaster import Forecaster, ForecastSource
from qraft.utils.log import info_event, warning_event

logger = logging.getLogger(__name__)


@dataclass
class DecisionPoint:
    index: int
    decision_bar: datetime
    execution_bar: datetime
    snapshot: MarketSnapshot


class DecideOrHoldResult(NamedTuple):
    decision: PolicyDecision
    status: str
    sigma: NDArray[np.floating] | None
    error_msg: str | None
    warning: BacktestWarning | None
    policy_inputs: PolicyInputs | None


def decide_or_hold(
    policy: PolicyProtocol,
    inputs: PolicyInputsProvider | None,
    state: PortfolioState,
    point: DecisionPoint,
    asset_order: list[str],
) -> DecideOrHoldResult:
    try:
        policy_inputs = (
            inputs.for_date(point.snapshot, point.index) if inputs is not None else None
        )
        decision = policy.decide(state, policy_inputs)
        status = getattr(decision.diagnostics, "status", "ok")
        sigma = _aligned_sigma(policy_inputs, asset_order)
        return DecideOrHoldResult(decision, status, sigma, None, None, policy_inputs)
    except (OptimizationFailure, RuntimeError, ValueError) as exc:
        bar = point.decision_bar
        warning_event(
            logger,
            "backtest.decision_failed",
            "Decision failed; holding current weights",
            decision_bar=bar,
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
        logger.debug(
            "Traceback for failed decision at %s:\n%s", bar, traceback.format_exc()
        )
        exc_str = f"{type(exc).__name__}: {exc}"
        warning = _make_warning(
            message="decision failed; holding current weights",
            bar=bar,
            exception=exc_str,
        )
        decision = PolicyDecision(
            asset_order, state.asset_weights, float(state.cash_weight), hold=True
        )
        status = getattr(exc, "status", "solver_error")
        return DecideOrHoldResult(decision, status, None, exc_str, warning, None)


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


def _charge(cash: float, amount: float, nav: float, label: str, bar: datetime) -> float:
    """Debit a realised cost from cash; absorb float-noise, fail on a real overdraft."""
    cash -= amount
    if cash >= 0.0 or cash > -1e-9 * max(abs(nav), 1.0):
        return max(cash, 0.0)
    raise ValueError(
        f"{label} at {bar} drove cash negative ({cash:.6g}); add a "
        "MinCashWeight buffer or allow_borrow."
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
    pending: Any,
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
    d_bar, decision, status, error_msg, sigma = pending
    before = PortfolioState(asset_order, prices, shares, cash)
    executed, shares, cash = execute_frictionless(
        decision, shares, cash, prices, asset_order
    )
    trade_cost = costs.trade_charge(executed, prices, nav_pre, sigma=sigma)
    cash = _charge(cash, trade_cost, nav_pre, "trade cost", bar)
    diagnostics = getattr(decision, "diagnostics", None)
    dropped_assets = tuple(
        getattr(drop, "asset", str(drop))
        for drop in getattr(diagnostics, "dropped_assets", ())
    )
    asset_diagnostics = tuple(getattr(diagnostics, "asset_diagnostics", ()))
    return (
        shares,
        cash,
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
            dropped_assets=dropped_assets,
            asset_diagnostics=asset_diagnostics,
        ),
    )


def decision_points(
    market: MarketData, schedule: RebalanceSchedule, warmup: int, step_size: int = 1
) -> list[DecisionPoint]:
    all_bars = market.trading_bars
    decision_bars = {d for d, _ in schedule.decision_steps(all_bars)}
    decision_points: list[DecisionPoint] = []

    for i, bar in enumerate(all_bars):
        if bar in decision_bars and i + 1 >= warmup:
            execution_bar = schedule.execution_bar(
                decision_bar=bar, trading_bars=all_bars
            )
            if execution_bar is None:  # skip as last one
                continue
            decision_points.append(
                DecisionPoint(
                    index=len(decision_points),
                    decision_bar=bar,
                    execution_bar=execution_bar,
                    snapshot=market.snapshot_at(
                        t=bar, t_next=execution_bar, step_size=step_size
                    ),
                )
            )
    return decision_points


def precompute_inputs(
    market: MarketData,
    schedule: RebalanceSchedule,
    warmup: int,
    *,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
    source: ForecastSource
    | PolicyInputsProvider
    | dict[datetime, PolicyInputs]
    | None = None,
    policy: PolicyInputRequirements | None = None,
    dtype: type = np.float64,
    step_size: int = 1,
) -> dict[datetime, PolicyInputs]:
    """Build and freeze policy inputs for the exact backtest decision schedule."""
    market.assert_backtest_safe()
    points = decision_points(market, schedule, warmup, step_size=step_size)
    if isinstance(source, dict):
        table = source
    elif isinstance(source, PolicyInputsProvider):
        table = {
            point.decision_bar: source.for_date(point.snapshot, point.index)
            for point in points
        }
    else:
        forecast_source = source if source is not None else forecaster
        if forecast_source is None:
            raise TypeError("precompute_inputs requires source or forecaster")
        if plan is None:
            raise TypeError("forecast-backed precompute_inputs requires plan")
        table = build_policy_input_table(
            (point.snapshot for point in points),
            forecast_source,
            plan=plan,
            policy=policy,
            dtype=dtype,
            market=market,
        )
    info_event(
        logger,
        "policy_inputs.completed",
        "Policy inputs precomputed",
        decisions=len(table),
        source_type=type(source).__name__
        if source is not None
        else type(forecaster).__name__,
    )
    return table


def run_backtest(
    market: MarketData,
    policy: PolicyProtocol,
    *,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    inputs: PolicyInputsProvider | None = None,
    initial_cash: float = 100.0,
    step_size: int = 1,
    costs: CostModel | None = None,
) -> BacktestResult:
    if costs is None:
        costs = CostModel.from_policy(policy)

    warmup = policy.min_history

    bars = market.trading_bars
    asset_order = list(market.universe.assets)
    points = decision_points(market, schedule, warmup, step_size=step_size)
    points_by_bar = {p.decision_bar: p for p in points}
    exec_to_decision = {p.execution_bar: p.decision_bar for p in points}

    shares = np.zeros(len(asset_order))
    cash = initial_cash
    pending = None  # (decision_bar, decision, status)
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

        # decide from information <= bar; queue for the next bar (after warm-up)
        if bar in points_by_bar:
            point = points_by_bar[bar]
            state = PortfolioState(asset_order, point.snapshot.prices_t, shares, cash)
            decision, status, sigma, error_msg, warning, policy_inputs = decide_or_hold(
                policy, inputs, state, point, asset_order
            )
            if policy_inputs is not None:
                invariance_drops.extend(getattr(policy_inputs, "invariance_drops", ()))
            if warning is not None:
                warnings_log.append(warning)
            pending = (bar, decision, status, error_msg, sigma)

        nav_dates.append(bar)
        nav.append(float(shares @ prices + cash))
        prev = bar

    if warmup and not periods:
        msg = f"No decisions: history never reached policy.min_history={warmup}."
        warning_event(logger, "backtest.no_decisions", msg, warmup=warmup)
        warnings_log.append(_make_warning(message=msg))

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
