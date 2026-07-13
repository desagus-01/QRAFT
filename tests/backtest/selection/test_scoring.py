import math

import pytest

from qraft.backtest.result import PerformanceSummary
from qraft.backtest.selection.scoring import (
    aggregate_scores,
    finite_score_summary,
    score_name,
    score_summary,
)


def _summary(**overrides) -> PerformanceSummary:
    values = dict(
        total_return=0.1,
        annualised_return=0.2,
        annualised_vol=0.3,
        sharpe=1.5,
        sortino=1.7,
        max_drawdown=-0.2,
        calmar=0.8,
        cvar=-0.04,
        hit_rate=0.6,
        avg_turnover=0.1,
        total_cost=2.0,
        n_periods=12,
        n_warnings=0,
        n_solver_failures=0,
        held_fraction=0.0,
    )
    values.update(overrides)
    return PerformanceSummary(**values)


def test_score_summary_keeps_higher_is_better_metrics_positive() -> None:
    assert score_summary(_summary(), "sharpe") == 1.5


def test_score_summary_negates_lower_is_better_cvar() -> None:
    assert score_summary(_summary(cvar=0.04), "cvar") == -0.04


def test_score_summary_uses_callable_as_is() -> None:
    def cvar_raw(summary: PerformanceSummary) -> float:
        return summary.cvar

    assert score_summary(_summary(cvar=0.04), cvar_raw) == 0.04
    assert score_name(cvar_raw) == "cvar_raw"


def test_finite_score_summary_excludes_non_finite_scores(caplog) -> None:
    assert finite_score_summary(_summary(sharpe=math.inf), "sharpe") is None
    assert "Excluding non-finite selection score" in caplog.text


def test_aggregate_scores_ignores_non_finite_values() -> None:
    assert aggregate_scores([1.0, math.nan, 3.0], "mean") == 2.0
    assert aggregate_scores([1.0, math.inf, 3.0], "median") == 2.0


def test_aggregate_scores_returns_nan_when_no_finite_values() -> None:
    assert math.isnan(aggregate_scores([math.nan, math.inf], "mean"))


def test_aggregate_scores_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="unsupported aggregation"):
        aggregate_scores([1.0], "mode")  # type: ignore[arg-type]
