from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from qraft.backtest.engine.schedule import DecisionPoint
from qraft.backtest.inputs import OptimizerInputsProvider
from qraft.backtest.result import BacktestWarning
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.optimization.optimization import OptimizationFailure
from qraft.construction.policies import PolicyDecision, PolicyProtocol
from qraft.construction.state import PortfolioState
from qraft.utils.log import warning_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PolicyStepResult:
    decision: PolicyDecision
    solver_status: str
    sigma: NDArray[np.floating] | None
    decision_error: str | None
    warning: BacktestWarning | None
    optimizer_inputs: OptimizerInputs | None


@dataclass(frozen=True, slots=True)
class PendingDecision:
    decision_bar: datetime
    decision: PolicyDecision
    solver_status: str
    decision_error: str | None
    sigma: NDArray[np.floating] | None
    view_diagnostics: Any = None


def decide_or_hold(
    policy: PolicyProtocol,
    inputs: OptimizerInputsProvider | None,
    state: PortfolioState,
    point: DecisionPoint,
    asset_order: list[str],
) -> PolicyStepResult:
    try:
        optimizer_inputs = (
            inputs.for_date(point.snapshot, point.index) if inputs is not None else None
        )
        decision = policy.decide(state, optimizer_inputs)
        status = getattr(decision.diagnostics, "status", "ok")
        sigma = _aligned_sigma(optimizer_inputs, asset_order)
        return PolicyStepResult(decision, status, sigma, None, None, optimizer_inputs)
    except (OptimizationFailure, RuntimeError, ValueError, cp.error.SolverError) as exc:
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
        warning = make_warning(
            message="decision failed; holding current weights",
            bar=bar,
            exception=exc_str,
        )
        decision = PolicyDecision(
            asset_order, state.asset_weights, float(state.cash_weight), hold=True
        )
        status = getattr(exc, "status", "solver_error")
        return PolicyStepResult(decision, status, None, exc_str, warning, None)


def make_warning(
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
    optimizer_inputs: OptimizerInputs | None,
    asset_order: list[str],
) -> NDArray[np.floating] | None:
    """Forecast per-asset sigma aligned to ``asset_order`` for the impact term."""
    if optimizer_inputs is None or optimizer_inputs.covariances is None:
        return None
    diag = np.sqrt(np.maximum(np.diag(optimizer_inputs.covariances[0]), 0.0))
    sig = dict(zip(optimizer_inputs.assets, diag))
    return np.array([sig.get(asset, 0.0) for asset in asset_order], dtype=float)
