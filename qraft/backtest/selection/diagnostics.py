from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from qraft.backtest.selection.results import CandidateResult
from qraft.backtest.selection.significance import (
    deflated_sharpe,
    prob_of_backtest_overfit,
)
from qraft.core import metrics
from qraft.core.cadence import infer_periods_per_year


def resolve_selection_periods_per_year(
    candidate_results: Sequence[CandidateResult], configured: float | None
) -> float:
    if configured is not None:
        return configured
    for candidate_result in candidate_results:
        if candidate_result.backtest is None:
            continue
        if candidate_result.backtest.periods_per_year is not None:
            return candidate_result.backtest.periods_per_year
        if len(candidate_result.backtest.nav_dates) >= 2:
            return infer_periods_per_year(candidate_result.backtest.nav_dates)
    failures = [
        candidate_result.failure
        for candidate_result in candidate_results
        if candidate_result.failure is not None
    ]
    if failures and len(failures) == len(candidate_results):
        reasons = "; ".join(
            f"{failure.params}: {failure.reason}" if failure.params else failure.reason
            for failure in failures
        )
        raise ValueError(
            "No successful candidates; cannot infer periods_per_year. "
            f"Candidate failures: {reasons}"
        )
    raise ValueError(
        "periods_per_year is required when no successful candidate can infer it."
    )


def trial_sharpes(
    candidate_results: Sequence[CandidateResult], periods_per_year: float | None
) -> list[float]:
    """Per-period Sharpe values from successful full-run candidates."""
    if periods_per_year is None:
        periods_per_year = resolve_selection_periods_per_year(candidate_results, None)
    return [
        candidate_result.summary.sharpe / math.sqrt(periods_per_year)
        for candidate_result in candidate_results
        if candidate_result.summary is not None
    ]


def compute_deflated_sharpe(
    returns: NDArray[np.floating], trial_sharpe_values: Sequence[float]
) -> float | None:
    if returns.size == 0 or len(trial_sharpe_values) < 2:
        return None
    return deflated_sharpe(returns, trial_sharpe_values)


def compute_pbo(candidate_results: Sequence[CandidateResult], n_blocks: int) -> float:
    block_returns = candidate_block_returns(candidate_results, n_blocks)
    if block_returns is None:
        raise ValueError("Block returns is None")
    return prob_of_backtest_overfit(block_returns)


def candidate_block_returns(
    candidate_results: Sequence[CandidateResult], n_blocks: int
) -> NDArray[np.floating] | None:
    """Build the ``(n_blocks, n_candidates)`` mean-return matrix PBO needs."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    successful_results = [
        candidate_result
        for candidate_result in candidate_results
        if candidate_result.backtest is not None
    ]
    if len(successful_results) < 2:
        raise ValueError("PBO needs at least 2 successful candidates")
    first_backtest = successful_results[0].backtest
    assert first_backtest is not None
    dates = first_backtest.nav_dates
    if len(dates) != len(first_backtest.nav):
        raise ValueError(
            f"Candidate 0 has {len(dates)} nav_dates but {len(first_backtest.nav)} NAV values"
        )
    for candidate_index, candidate_result in enumerate(successful_results[1:], start=1):
        assert candidate_result.backtest is not None
        candidate_dates = candidate_result.backtest.nav_dates
        if len(candidate_dates) != len(candidate_result.backtest.nav):
            raise ValueError(
                f"Candidate {candidate_index} has {len(candidate_dates)} nav_dates but "
                f"{len(candidate_result.backtest.nav)} NAV values"
            )
        if len(candidate_dates) != len(dates):
            raise ValueError("Candidate NAV dates do not match: length differs")
        for date_index, (expected, actual) in enumerate(
            zip(dates, candidate_dates, strict=True)
        ):
            if actual != expected:
                raise ValueError(
                    "Candidate NAV dates do not match "
                    f"at index {date_index}: expected {expected}, got {actual}"
                )
    if len(dates) < 2 * n_blocks:
        raise ValueError("Not enough data.")

    edges = np.linspace(0, len(dates), n_blocks + 1).astype(int)
    performance = np.zeros((n_blocks, len(successful_results)))
    for block_index in range(n_blocks):
        start_index, end_index = int(edges[block_index]), int(edges[block_index + 1])
        if end_index - start_index < 2:
            raise ValueError("Not enough data.")
        start, end = dates[start_index], dates[end_index - 1]
        for candidate_index, candidate_result in enumerate(successful_results):
            assert candidate_result.backtest is not None
            returns = metrics.returns_from_nav(
                candidate_result.backtest.window(start, end).nav
            )
            performance[block_index, candidate_index] = (
                float(returns.mean()) if returns.size else 0.0
            )
    return performance
