from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.selection.results import CandidateResult, PolicyParams
from qraft.backtest.selection.scoring import (
    Agg,
    Scorer,
    aggregate_scores,
    returns_for_range,
    score_summary,
    summary_from_returns,
)
from qraft.backtest.selection.splits import DateRange
from qraft.core.configs import SelectionMetric


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Full-grid backtest artifact reused by reports and tuning."""

    candidate_results: tuple[CandidateResult, ...]
    dates: list[datetime]
    backtest_config: BacktestConfig
    metric: SelectionMetric
    risk_free_rate: float = 0.0
    scorer: Scorer | None = None

    def oos_scores(
        self,
        windows_per_candidate: Mapping[PolicyParams, Sequence[DateRange]],
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
                    window_scores.append(
                        score_summary(summary, self.metric, self.scorer)
                    )
            if window_scores:
                scores[candidate.params] = aggregate_scores(window_scores, agg)
        return scores

    def full_sample_scores(self) -> dict[PolicyParams, float]:
        return {
            candidate.params: score_summary(candidate.summary, self.metric, self.scorer)
            for candidate in self.candidate_results
            if candidate.failure is None and candidate.summary is not None
        }
