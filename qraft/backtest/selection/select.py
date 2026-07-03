from collections.abc import Sequence

from qraft.backtest.selection.results import CandidateResult, SelectionReport
from qraft.backtest.selection.scoring import ScoreSpec, finite_score_summary, score_name


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


def _objective(result: CandidateResult, score: ScoreSpec) -> float:
    """Score one eligible candidate; higher is better."""
    summary = result.summary
    assert summary is not None
    value = finite_score_summary(summary, score)
    return float("-inf") if value is None else value


def select_candidate(
    results: Sequence[CandidateResult],
    *,
    max_held_fraction: float = 0.5,
    score: ScoreSpec = "sharpe",
) -> SelectionReport:
    eligible = [
        r
        for r in results
        if _eligible(r, max_held_fraction) and _objective(r, score) != float("-inf")
    ]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda r: (_objective(r, score), _conservatism(r)),
        ).params
    rule = f"max[{score_name(score)}] | exclude failed & held_fraction>{max_held_fraction:g}"
    return SelectionReport(
        candidates=tuple(results), selected_params=selected, rule=rule
    )
