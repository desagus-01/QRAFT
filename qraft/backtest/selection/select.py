from collections.abc import Sequence

from qraft.backtest.selection.results import CandidateResult, SelectionReport
from qraft.backtest.selection.scoring import Scorer, score_summary
from qraft.core.configs import SelectionMetric


def _eligible(result: CandidateResult, max_held_fraction: float) -> bool:
    """Drop failed candidates and ones that barely traded (degenerate)."""
    return (
        result.failure is None
        and result.summary is not None
        and result.summary.held_fraction <= max_held_fraction
    )


def _conservatism(result: CandidateResult) -> float:
    """Tie-break toward the more risk-averse (higher risk_aversion) candidate."""
    return float(result.params.as_dict().get("risk_aversion", 0.0))


def _objective(
    result: CandidateResult, metric: SelectionMetric, score: Scorer | None
) -> float:
    """Score one eligible candidate: the custom scorer if given, else the metric."""
    summary = result.summary
    assert summary is not None
    return score_summary(summary, metric, score)


def select_candidate(
    results: Sequence[CandidateResult],
    *,
    metric: SelectionMetric = "sharpe",
    max_held_fraction: float = 0.5,
    score: Scorer | None = None,
) -> SelectionReport:
    eligible = [r for r in results if _eligible(r, max_held_fraction)]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda r: (_objective(r, metric, score), _conservatism(r)),
        ).params
    objective = "custom" if score is not None else f"max[{metric}]"
    rule = f"{objective} | exclude failed & held_fraction>{max_held_fraction:g}"
    return SelectionReport(
        candidates=tuple(results), selected_params=selected, rule=rule
    )
