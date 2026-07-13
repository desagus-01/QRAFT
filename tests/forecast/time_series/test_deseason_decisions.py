import numpy as np
import polars as pl

from qraft.forecast.pipelines.preprocess import deseason_pipeline
from qraft.forecast.time_series.preprocessing.decisions import (
    MAX_SEASONAL_HARMONICS,
    _expand_period_to_harmonics,
    deseason_decision_rule,
)
from qraft.forecast.time_series.tests.seasonality import SeasonalityPeriodTest
from qraft.forecast.time_series.tests.types import HypTestRes


def _seasonality_test(period: str, evidence: bool) -> SeasonalityPeriodTest:
    return SeasonalityPeriodTest(
        seasonal_period=period,
        seasonal_frequency_radian=2.0 * np.pi / 63.0,
        evidence_of_seasonality=evidence,
        res=HypTestRes(
            stat=1.0,
            p_val=0.01 if evidence else 0.99,
            sign_lvl=0.05,
            null=f"No {period} seasonality",
            reject_null=evidence,
            desc="",
        ),
    )


def test_expand_quarterly_period_uses_low_order_harmonics_by_default() -> None:
    harmonics = _expand_period_to_harmonics("quarterly")

    assert len(harmonics) == MAX_SEASONAL_HARMONICS
    np.testing.assert_allclose(
        [omega for _, omega in harmonics],
        [2.0 * np.pi * k / 63.0 for k in range(1, MAX_SEASONAL_HARMONICS + 1)],
    )


def test_deseason_decision_rule_does_not_expand_quarterly_to_full_basis() -> None:
    decision = deseason_decision_rule({"MTX": [_seasonality_test("quarterly", True)]})

    assert [label for label, _ in decision["MTX"]] == [
        "quarterly_h1",
        "quarterly_h2",
        "quarterly_h3",
    ]


def test_deseason_decision_rule_keeps_weak_seasonality_unadjusted() -> None:
    decision = deseason_decision_rule({"MTX": [_seasonality_test("quarterly", False)]})

    assert decision["MTX"] == []


def test_deseason_pipeline_returns_date_only_for_empty_assets() -> None:
    data = pl.DataFrame(
        {
            "date": [1, 2, 3],
            "asset_a": [1.0, 2.0, 3.0],
        }
    )

    res = deseason_pipeline(data=data, assets=[], include_diagnostics=True)

    assert res.decision == {}
    assert res.inverse_spec == {}
    assert res.all_tests == {}
    assert res.updated_data.columns == ["date"]
