from qraft.forecast.time_series.preprocessing.decisions import detrend_decision_rule
from qraft.forecast.time_series.selection.trend import AssetTrendDiagnostic


def test_detrend_decision_rule_defaults_inconclusive_to_first_difference():
    decision = detrend_decision_rule(
        detrend_res={"asset_a": AssetTrendDiagnostic(asset="asset_a")},
        assets=["asset_a"],
    )

    assert decision["asset_a"].kind == "difference"
    assert decision["asset_a"].order == 1
