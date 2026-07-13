import pytest

from qraft.backtest.selection import (
    CandidateFailure,
    CandidateResult,
    PolicyCandidate,
    PolicyParams,
    SelectionReport,
)
from qraft.backtest.selection.diagnostics import resolve_selection_periods_per_year
from qraft.construction.policies import EqualWeightPolicy


def test_policy_params_sort_and_round_trip_values():
    params = PolicyParams.of(risk_aversion=2.0, alpha=0.05, label="base")

    assert params.values == (
        ("alpha", 0.05),
        ("label", "base"),
        ("risk_aversion", 2.0),
    )
    assert params.as_dict() == {
        "alpha": 0.05,
        "label": "base",
        "risk_aversion": 2.0,
    }


def test_policy_params_string_is_stable():
    params = PolicyParams.of(risk_aversion=2.0, alpha=0.05, label="base")

    assert str(params) == "alpha=0.05, label=base, risk_aversion=2"


def test_policy_params_bool_reflects_values():
    assert not PolicyParams.of()
    assert PolicyParams.of(risk_aversion=1.0)


def test_selection_result_records_are_importable_and_generic():
    params = PolicyParams.of(risk_aversion=1.0)
    policy = EqualWeightPolicy(target_cash_weight=0.0)
    candidate = PolicyCandidate(params=params, policy=policy)
    failure = CandidateFailure(params=params, reason="infeasible")
    result = CandidateResult(params=params, failure=failure)
    report = SelectionReport(candidates=(result,), selected_params=None, rule="none")

    assert candidate.params == params
    assert candidate.policy is policy
    assert report.candidates == (result,)
    assert report.candidates[0].failure == failure


def test_periods_per_year_error_reports_all_candidate_failures():
    params = PolicyParams.of(risk_aversion=1.0)
    failure = CandidateFailure(params=params, reason="trade cost drove cash negative")
    result = CandidateResult(params=params, failure=failure)

    with pytest.raises(ValueError, match="trade cost drove cash negative"):
        resolve_selection_periods_per_year([result], None)
