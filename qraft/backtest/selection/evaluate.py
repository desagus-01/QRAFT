from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from qraft.backtest.inputs import PolicyInputsProvider, PrecomputedInputsProvider
from qraft.backtest.market import MarketData
from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.schedule import RebalanceSchedule
from qraft.backtest.selection.candidates import expand_candidates
from qraft.backtest.selection.results import (
    CandidateFailure,
    CandidateResult,
    PolicyCandidate,
)
from qraft.backtest.simulator import precompute_inputs, run_backtest
from qraft.construction.policies import PolicyProtocol
from qraft.core.configs import BacktestConfig

logger = logging.getLogger(__name__)


def evaluate_candidates(
    candidates: Sequence[PolicyCandidate],
    market: MarketData,
    inputs: PolicyInputsProvider,
    *,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    step_size: int = 1,
    initial_cash: float = 100.0,
    periods_per_year: float = 252.0,
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
            logger.info("candidate %s: sharpe=%.3f", candidate.params, summary.sharpe)
        except Exception as exc:  # isolate: one bad candidate must not kill the sweep
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("candidate %s failed: %s", candidate.params, reason)
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
    provider: PolicyInputsProvider,
    *,
    schedule: RebalanceSchedule = RebalanceSchedule(),
    step_size: int = 1,
    initial_cash: float = 100.0,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> tuple[CandidateResult, ...]:
    """Expand candidates, precompute moments ONCE, then evaluate them all."""
    candidates = expand_candidates(base_policy, grid)
    warmup = _shared_warmup(candidates)
    table = precompute_inputs(market, schedule, provider, warmup, step_size=step_size)
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
    provider: PolicyInputsProvider,
    backtest_config: BacktestConfig,
    risk_free_rate: float,
) -> tuple[tuple[CandidateResult, ...], list[datetime]]:
    """Expand a candidate grid, precompute shared inputs, and evaluate once."""
    candidates = expand_candidates(base_policy, grid)
    warmup = _shared_warmup(candidates)
    table = precompute_inputs(
        market,
        backtest_config.schedule,
        provider,
        warmup,
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
    return candidate_results, sorted(table)


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
