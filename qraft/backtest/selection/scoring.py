from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.execution import BacktestResult
from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.selection.results import CandidateResult, PolicyParams
from qraft.backtest.selection.splits import DateRange
from qraft.core import metrics
from qraft.core.configs import SelectionMetric

Agg = Literal["mean", "median"]
Scorer = Callable[[PerformanceSummary], float]

_LOWER_IS_BETTER: frozenset[SelectionMetric] = frozenset({"cvar"})


def score_summary(
    summary: PerformanceSummary,
    metric: SelectionMetric,
    scorer: Scorer | None = None,
) -> float:
    """Signed score so the best candidate is always the maximum."""
    if scorer is not None:
        return float(scorer(summary))
    value = float(getattr(summary, metric))
    return -value if metric in _LOWER_IS_BETTER else value


def aggregate_scores(scores: Sequence[float], agg: Agg) -> float:
    if not scores:
        return float("nan")
    values = np.asarray(scores, dtype=float)
    match agg:
        case "mean":
            return float(np.mean(values))
        case "median":
            return float(np.median(values))
    raise ValueError(f"unsupported aggregation: {agg}")


def returns_for_range(
    backtest: BacktestResult,
    date_range: DateRange,
) -> NDArray[np.floating]:
    """Return period returns for one inclusive backtest date range."""
    start, end = date_range
    return metrics.returns_from_nav(backtest.window(start, end).nav)


def returns_for_ranges(
    backtest: BacktestResult,
    ranges: Sequence[DateRange],
) -> NDArray[np.floating]:
    """Return concatenated period returns for multiple date ranges."""
    return_segments = [
        return_segment
        for date_range in ranges
        if (return_segment := returns_for_range(backtest, date_range)).size
    ]
    return (
        np.concatenate(return_segments) if return_segments else np.empty(0, dtype=float)
    )


def summary_from_returns(
    returns: NDArray[np.floating],
    backtest_config: BacktestConfig,
    risk_free_rate: float,
    *,
    policy_name: str = "selection_returns",
) -> PerformanceSummary | None:
    """Build a synthetic NAV from returns so common summary metrics can score it."""
    if returns.size < 1:
        return None
    nav = np.concatenate(
        [
            [backtest_config.initial_cash],
            backtest_config.initial_cash * np.cumprod(1.0 + returns),
        ]
    )
    nav_dates = [datetime(2000, 1, 1) + timedelta(days=i) for i in range(nav.size)]
    synthetic = BacktestResult(policy_name, [], nav_dates, nav, [])
    return PerformanceSummary.from_backtest(
        synthetic,
        active_only=False,
        periods_per_year=backtest_config.periods_per_year,
        risk_free_rate=risk_free_rate,
    )


def score_candidate_range(
    candidate: CandidateResult,
    date_range: DateRange,
    backtest_config: BacktestConfig,
    risk_free_rate: float,
) -> CandidateResult:
    """Re-score a candidate over one leakage-free date range."""
    if candidate.failure is not None or candidate.backtest is None:
        return CandidateResult(params=candidate.params, failure=candidate.failure)
    windowed = candidate.backtest.window(date_range[0], date_range[1])
    summary = PerformanceSummary.from_backtest(
        windowed,
        active_only=False,
        periods_per_year=backtest_config.periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    return CandidateResult(params=candidate.params, summary=summary, backtest=windowed)


def score_candidate_ranges(
    candidate: CandidateResult,
    ranges: Sequence[DateRange],
    backtest_config: BacktestConfig,
    risk_free_rate: float,
) -> CandidateResult:
    """Re-score a candidate over a union of leakage-free date ranges."""
    if candidate.failure is not None or candidate.backtest is None:
        return CandidateResult(params=candidate.params, failure=candidate.failure)
    summary = summary_from_returns(
        returns_for_ranges(candidate.backtest, ranges),
        backtest_config,
        risk_free_rate,
    )
    return CandidateResult(params=candidate.params, summary=summary)


def find_candidate(
    results: Sequence[CandidateResult], params: PolicyParams
) -> CandidateResult:
    return next(result for result in results if result.params == params)
