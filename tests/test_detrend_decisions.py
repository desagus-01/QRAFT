import numpy as np
import polars as pl

from qraft.forecast.pipelines.preprocess import detrend_pipeline
from qraft.forecast.time_series.preprocessing.decisions import detrend_decision_rule
from qraft.forecast.time_series.selection.trend import AssetTrendDiagnostic


def test_detrend_decision_rule_defaults_inconclusive_to_first_difference():
    decision = detrend_decision_rule(
        detrend_res={"asset_a": AssetTrendDiagnostic(asset="asset_a")},
        assets=["asset_a"],
    )

    assert decision["asset_a"].kind == "difference"
    assert decision["asset_a"].order == 1


def test_detrend_pipeline_returns_date_only_for_empty_assets() -> None:
    data = pl.DataFrame(
        {
            "date": [1, 2, 3],
            "asset_a": [1.0, 2.0, 3.0],
        }
    )

    res = detrend_pipeline(
        data=data,
        prob=np.full(data.height, 1.0 / data.height),
        assets=[],
        include_diagnostics=True,
    )

    assert res.decision == {}
    assert res.inverse_spec == {}
    assert res.all_tests == {}
    assert res.updated_data.columns == ["date"]
