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
from qraft.backtest.inputs import (
    PolicyInputRequirements,
    PolicyInputsProvider,
    policy_risk_source,
)
from qraft.backtest.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.construction.inputs import (
    build_policy_input_table_from_forecast_run,
    forecast_policy_input_table,
)
from qraft.construction.optimization.moments import PolicyInputConfig, PolicyInputs
from qraft.construction.optimization.optimization import OptimizationFailure
from qraft.construction.policies import PolicyDecision, PolicyProtocol
from qraft.construction.state import PortfolioState
from qraft.core.configs import (
    DEFAULT_FORECAST_PROVIDER_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    ForecastProviderConfig,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.forecast.run import ForecastRecipeHistory, simulate_forecast_paths
from qraft.core.snapshot import MarketSnapshot

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
        return DecideOrHoldResult(decision, status, sigma, None, None)
    except (OptimizationFailure, RuntimeError, ValueError) as exc:
        bar = point.decision_bar
        logger.warning("decision at %s failed (%s); holding.", bar, exc)
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
            asset_order, state.asset_weights, float(state.cash_weight)
        )
        status = getattr(exc, "status", "solver_error")
        return DecideOrHoldResult(decision, status, None, exc_str, warning)


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
    provider: PolicyInputsProvider,
    warmup: int,
    step_size: int = 1,
) -> dict[datetime, PolicyInputs]:
    """Run the forecast->PolicyInputs pipeline once and freeze it by date.

    Drives ``provider`` over the exact decision snapshots (and indices) that
    ``run_backtest`` will request, so the captured moments match a live run
    bar-for-bar. Wrap the result in ``PrecomputedInputsProvider`` and reuse it
    read-only across every candidate policy.

    Moments depend only on the input layer (forecasts/history/cash), never on
    policy-layer hyperparameters (risk aversion, turnover, min-cash, cost
    weights) -- so one table serves the whole policy-layer grid. The table is
    valid only for candidates sharing this ``provider``, ``warmup``,
    ``step_size`` and ``schedule``; any input-layer change needs its own table.
    """
    points = decision_points(market, schedule, warmup, step_size=step_size)
    table = {
        point.decision_bar: provider.for_date(point.snapshot, point.index)
        for point in points
    }
    logger.info("precompute_inputs: cached %d decision date(s)", len(table))
    return table


def precompute_forecast_inputs(
    market: MarketData,
    schedule: RebalanceSchedule,
    input_config: PolicyInputConfig,
    warmup: int,
    *,
    policy: PolicyInputRequirements | None = None,
    provider_config: ForecastProviderConfig = DEFAULT_FORECAST_PROVIDER_CONFIG,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
    dtype: type = np.float64,
    step_size: int = 1,
) -> dict[datetime, PolicyInputs]:
    """Forecast once for the decision schedule, then freeze inputs by date."""
    points = decision_points(market, schedule, warmup, step_size=step_size)
    table = forecast_policy_input_table(
        (point.snapshot for point in points),
        input_config=input_config,
        risk_source=policy_risk_source(input_config, policy),
        provider_config=provider_config,
        pipeline_config=pipeline_config,
        simulation_config=simulation_config,
        dtype=dtype,
    )
    logger.info("precompute_forecast_inputs: cached %d decision date(s)", len(table))
    return table


def precompute_inputs_from_recipe_history(
    market: MarketData,
    schedule: RebalanceSchedule,
    input_config: PolicyInputConfig,
    recipe_history: ForecastRecipeHistory,
    warmup: int,
    *,
    policy: PolicyInputRequirements | None = None,
    seed: int | None = None,
    simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
    dtype: type = np.float64,
    step_size: int = 1,
) -> dict[datetime, PolicyInputs]:
    """Simulate forecasts from fixed recipes, then freeze policy inputs by date."""
    points = decision_points(market, schedule, warmup, step_size=step_size)
    run = simulate_forecast_paths(
        market,
        recipe_history,
        min_history=warmup,
        forecast_cadence=schedule.cadence,
        seed=seed,
        simulation_config=simulation_config,
    )
    table = build_policy_input_table_from_forecast_run(
        (point.snapshot for point in points),
        run,
        input_config=input_config,
        risk_source=policy_risk_source(input_config, policy),
        dtype=dtype,
    )
    logger.info(
        "precompute_inputs_from_recipe_history: cached %d decision date(s)",
        len(table),
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

    warnings_log: list[BacktestWarning] = []

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
            decision, status, sigma, error_msg, warning = decide_or_hold(
                policy, inputs, state, point, asset_order
            )
            if warning is not None:
                warnings_log.append(warning)
            pending = (bar, decision, status, error_msg, sigma)

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
        periods_per_year=market.config.periods_per_year,
    )
