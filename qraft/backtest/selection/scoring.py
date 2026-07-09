from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.result import BacktestResult, PerformanceSummary, summary_from_nav
from qraft.backtest.selection.results import CandidateResult, PolicyParams
from qraft.backtest.selection.splits import DateRange
from qraft.core import metrics
from qraft.core.configs import SelectionMetric

Agg = Literal["mean", "median"]
Scorer = Callable[[PerformanceSummary], float]
ScoreSpec = SelectionMetric | Scorer

_LOWER_IS_BETTER: frozenset[SelectionMetric] = frozenset({"cvar"})
logger = logging.getLogger(__name__)


def score_summary(
    summary: PerformanceSummary,
    score: ScoreSpec = "sharpe",
) -> float:
    """Signed score so the best candidate is always the maximum."""
    if callable(score):
        return float(score(summary))
    value = float(getattr(summary, score))
    return -value if score in _LOWER_IS_BETTER else value


def finite_score_summary(
    summary: PerformanceSummary,
    score: ScoreSpec = "sharpe",
) -> float | None:
    value = score_summary(summary, score)
    if np.isfinite(value):
        return value
    logger.warning(
        "Excluding non-finite selection score for score=%s", _score_name(score)
    )
    return None


def score_name(score: ScoreSpec) -> str:
    return _score_name(score)


def _score_name(score: ScoreSpec) -> str:
    return getattr(score, "__name__", "custom") if callable(score) else score


def aggregate_scores(scores: Sequence[float], agg: Agg) -> float:
    if not scores:
        return float("nan")
    values = np.asarray([score for score in scores if np.isfinite(score)], dtype=float)
    if values.size == 0:
        logger.warning(
            "aggregate_scores is NaN: all candidate/window scores are non-finite."
        )
        return float("nan")
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
    periods_per_year: float | None = None,
) -> PerformanceSummary | None:
    """Build a summary directly from a synthetic NAV path."""
    if returns.size < 1:
        return None
    nav = np.concatenate(
        [
            [backtest_config.initial_cash],
            backtest_config.initial_cash * np.cumprod(1.0 + returns),
        ]
    )
    return summary_from_nav(
        nav,
        periods_per_year=periods_per_year or backtest_config.periods_per_year,
        risk_free_rate=risk_free_rate,
        n_periods=int(returns.size),
    )


def score_candidate_range(
    candidate: CandidateResult,
    date_range: DateRange,
    backtest_config: BacktestConfig,
    risk_free_rate: float,
    *,
    periods_per_year: float | None = None,
) -> CandidateResult:
    """Re-score a candidate over one leakage-free date range."""
    if candidate.failure is not None or candidate.backtest is None:
        return CandidateResult(params=candidate.params, failure=candidate.failure)
    windowed = candidate.backtest.window(date_range[0], date_range[1])
    summary = PerformanceSummary.from_backtest(
        windowed,
        active_only=False,
        periods_per_year=periods_per_year or backtest_config.periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    return CandidateResult(params=candidate.params, summary=summary, backtest=windowed)


def score_candidate_ranges(
    candidate: CandidateResult,
    ranges: Sequence[DateRange],
    backtest_config: BacktestConfig,
    risk_free_rate: float,
    *,
    periods_per_year: float | None = None,
) -> CandidateResult:
    """Re-score a candidate over a union of leakage-free date ranges."""
    if candidate.failure is not None or candidate.backtest is None:
        return CandidateResult(params=candidate.params, failure=candidate.failure)
    summary = summary_from_returns(
        returns_for_ranges(candidate.backtest, ranges),
        backtest_config,
        risk_free_rate,
        periods_per_year=periods_per_year,
    )
    return CandidateResult(params=candidate.params, summary=summary)


def find_candidate(
    results: Sequence[CandidateResult], params: PolicyParams
) -> CandidateResult:
    return next(result for result in results if result.params == params)
