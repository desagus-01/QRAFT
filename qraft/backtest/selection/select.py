from collections.abc import Sequence
from typing import Literal

from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.selection.results import CandidateResult, SelectionReport

_METRIC = Literal[
    "total_return",
    "annualised_return",
    "annualised_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "cvar",
    "hit_rate",
    "avg_turnover",
    "total_cost",
]

_LOWER_IS_BETTER: frozenset[_METRIC] = frozenset({"cvar", "annualised_vol"})


def _score(summary: PerformanceSummary, metric: _METRIC) -> float:
    """Signed score so the best candidate is always the maximum."""
    value = float(getattr(summary, metric))
    return -value if metric in _LOWER_IS_BETTER else value


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


def select_candidate(
    results: Sequence[CandidateResult],
    *,
    metric: _METRIC = "sharpe",
    max_held_fraction: float = 0.5,
) -> SelectionReport:
    """Pick the best eligible candidate by a realised metric.

    Failures and degenerate candidates (``held_fraction`` above the threshold)
    are excluded; ties break toward the more conservative candidate. Returns a
    report with ``selected_params=None`` when nothing is eligible.
    """
    eligible = [r for r in results if _eligible(r, max_held_fraction)]
    selected = None
    if eligible:
        best = max(
            eligible,
            key=lambda r: (_score(r.summary, metric), _conservatism(r)),
        )
        selected = best.params
    rule = f"max[{metric}] | exclude failed & held_fraction>{max_held_fraction:g}"
    return SelectionReport(
        candidates=tuple(results), selected_params=selected, rule=rule
    )
