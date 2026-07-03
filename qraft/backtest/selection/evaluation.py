from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.selection.results import CandidateResult, PolicyParams
from qraft.backtest.selection.scoring import (
    Agg,
    ScoreSpec,
    aggregate_scores,
    finite_score_summary,
    returns_for_range,
    summary_from_returns,
)
from qraft.backtest.selection.splits import DateRange


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Full-grid backtest artifact reused by reports and tuning.

    Stores only what the grid backtest actually needs (raw results + config).
    Scoring parameters are supplied at query time so the same artifact serves
    every metric or custom scorer without cache invalidation.
    """

    candidate_results: tuple[CandidateResult, ...]
    dates: list[datetime]
    backtest_config: BacktestConfig
    risk_free_rate: float = 0.0

    def oos_scores(
        self,
        windows_per_candidate: Mapping[PolicyParams, Sequence[DateRange]],
        score: ScoreSpec = "sharpe",
        agg: Agg = "mean",
    ) -> dict[PolicyParams, float]:
        scores: dict[PolicyParams, float] = {}
        for candidate in self.candidate_results:
            if candidate.failure is not None or candidate.backtest is None:
                continue
            windows = windows_per_candidate.get(candidate.params, ())
            window_scores: list[float] = []
            for window in windows:
                returns = returns_for_range(candidate.backtest, window)
                if not returns.size:
                    continue
                summary = summary_from_returns(
                    returns,
                    self.backtest_config,
                    self.risk_free_rate,
                )
                if summary is not None:
                    value = finite_score_summary(summary, score)
                    if value is not None:
                        window_scores.append(value)
            if window_scores:
                scores[candidate.params] = aggregate_scores(window_scores, agg)
        return scores

    def full_sample_scores(
        self,
        score: ScoreSpec = "sharpe",
    ) -> dict[PolicyParams, float]:
        scores: dict[PolicyParams, float] = {}
        for candidate in self.candidate_results:
            if candidate.failure is not None or candidate.summary is None:
                continue
            value = finite_score_summary(candidate.summary, score)
            if value is not None:
                scores[candidate.params] = value
        return scores
