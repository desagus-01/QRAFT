from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, TypeAlias

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.inputs import PolicyInputsProvider, PrecomputedInputsProvider
from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.selection.candidates import expand_candidates
from qraft.backtest.selection.evaluation import CandidateEvaluation
from qraft.backtest.selection.results import (
    CandidateFailure,
    CandidateResult,
    PolicyCandidate,
)
from qraft.backtest.simulator import (
    precompute_inputs,
    run_backtest,
)
from qraft.construction.optimization.inputs import InputPlan, PolicyInputs
from qraft.construction.policies import PolicyProtocol
from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.forecast.forecaster import Forecaster, ForecastSource
from qraft.utils.log import debug_event, info_event, warning_event

SelectionInputSource: TypeAlias = (
    ForecastSource | PolicyInputsProvider | dict[datetime, PolicyInputs]
)

logger = logging.getLogger(__name__)


def evaluate_candidates(
    candidates: Sequence[PolicyCandidate],
    market: MarketData,
    inputs: PolicyInputsProvider,
    *,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    step_size: int = 1,
    initial_cash: float = 100.0,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
) -> tuple[CandidateResult, ...]:
    """Backtest each candidate against a SHARED inputs provider and score it.

    One forecast pass is shared across the whole sweep (the caller passes a
    precomputed provider). A candidate that raises becomes a CandidateFailure
    rather than aborting the run; a candidate that merely degenerates (all-cash,
    many solver holds) still returns a CandidateResult -- that shows up in the
    summary (held_fraction / n_solver_failures), not as a failure.
    """
    results: list[CandidateResult] = []
    for candidate in candidates:
        try:
            backtest = run_backtest(
                market,
                candidate.policy,
                schedule=schedule,
                inputs=inputs,
                initial_cash=initial_cash,
                step_size=step_size,
            )
            summary = PerformanceSummary.from_backtest(
                backtest,
                periods_per_year=periods_per_year,
                risk_free_rate=risk_free_rate,
            )
            results.append(
                CandidateResult(
                    params=candidate.params, summary=summary, backtest=backtest
                )
            )
            debug_event(
                logger,
                "validation.candidate_completed",
                "Candidate evaluation completed",
                params=candidate.params,
                sharpe=f"{summary.sharpe:.3f}",
            )
        except Exception as exc:  # isolate: one bad candidate must not kill the sweep
            reason = f"{type(exc).__name__}: {exc}"
            warning_event(
                logger,
                "validation.candidate_failed",
                "Candidate evaluation failed",
                params=candidate.params,
                reason=reason,
            )
            results.append(
                CandidateResult(
                    params=candidate.params,
                    failure=CandidateFailure(candidate.params, reason),
                )
            )
    return tuple(results)


def run_selection_window(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[Any]],
    *,
    source: SelectionInputSource | None = None,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    step_size: int = 1,
    initial_cash: float = 100.0,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
) -> tuple[CandidateResult, ...]:
    """Expand candidates, precompute moments ONCE, then evaluate them all."""
    if source is None and forecaster is None:
        raise TypeError("run_selection_window requires source or forecaster")
    candidates = expand_candidates(base_policy, grid)
    warmup = _shared_warmup(candidates)
    table = precompute_inputs(
        market,
        schedule,
        warmup,
        forecaster=forecaster,
        plan=plan,
        source=source,
        policy=base_policy,
        step_size=step_size,
    )
    shared = PrecomputedInputsProvider(table)
    return evaluate_candidates(
        candidates,
        market,
        shared,
        schedule=schedule,
        step_size=step_size,
        initial_cash=initial_cash,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )


def evaluate_candidate_grid(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[Any]],
    backtest_config: BacktestConfig,
    risk_free_rate: float,
    source: SelectionInputSource | None = None,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
) -> CandidateEvaluation:
    """Expand a candidate grid, precompute shared inputs, and evaluate once."""
    if source is None and forecaster is None:
        raise TypeError("evaluate_candidate_grid requires source or forecaster")
    candidates = expand_candidates(base_policy, grid)
    warmup = _shared_warmup(candidates)
    info_event(
        logger,
        "validation.started",
        "Validation candidate evaluation started",
        candidates=len(candidates),
        grid_keys=tuple(sorted(grid)),
        warmup=warmup,
        source_type=type(source).__name__
        if source is not None
        else type(forecaster).__name__,
    )
    table = precompute_inputs(
        market,
        backtest_config.schedule,
        warmup,
        forecaster=forecaster,
        plan=plan,
        source=source,
        policy=base_policy,
    )
    info_event(
        logger,
        "validation.candidates_precomputed",
        "Validation policy inputs precomputed",
        candidates=len(candidates),
        decisions=len(table),
    )
    shared = PrecomputedInputsProvider(table)
    candidate_results = evaluate_candidates(
        candidates,
        market,
        shared,
        schedule=backtest_config.schedule,
        initial_cash=backtest_config.initial_cash,
        periods_per_year=backtest_config.periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    failures = sum(1 for result in candidate_results if result.failure is not None)
    info_event(
        logger,
        "validation.completed",
        "Validation candidate evaluation completed",
        candidates=len(candidate_results),
        failures=failures,
        decisions=len(table),
    )
    return CandidateEvaluation(
        candidate_results=candidate_results,
        dates=sorted(table),
        backtest_config=backtest_config,
        risk_free_rate=risk_free_rate,
    )


def _shared_warmup(candidates: Sequence[PolicyCandidate]) -> int:
    """Overlays leave min_history untouched, so every candidate shares one warm-up
    -- which is exactly what the single shared inputs table requires."""
    warmups = {candidate.policy.min_history for candidate in candidates}
    if len(warmups) != 1:
        raise ValueError(
            f"candidates must share one min_history for a shared inputs table; "
            f"got {sorted(warmups)}"
        )
    return warmups.pop()
