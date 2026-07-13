import numpy as np
import pytest

from qraft.risk.feature_selection import StepResult, forward_regression


def test_forward_regression_r2_selects_predictive_feature_first() -> None:
    x1 = np.linspace(-2.0, 2.0, 30)
    x2 = np.sin(x1)
    x3 = np.cos(x1)
    y = (3.0 * x1 + 0.1 * x2).reshape(-1, 1)
    x = np.column_stack([x1, x2, x3])

    result = forward_regression(y, x, ["strong", "weak", "noise"], criterion="r2")

    assert result.selected_features[0] == "strong"
    assert result.steps[0].feature_added == "strong"
    assert result.final_model.r_squared > 0.99


def test_forward_regression_aic_stops_when_next_feature_does_not_improve() -> None:
    x1 = np.linspace(-2.0, 2.0, 40)
    x2 = np.ones_like(x1)
    y = (2.0 * x1).reshape(-1, 1)
    x = np.column_stack([x1, x2])

    result = forward_regression(y, x, ["strong", "constant"], criterion="aic")

    assert result.selected_features == ["strong"]
    assert result.n_selected == 1


def test_forward_regression_p_value_stops_above_threshold() -> None:
    x1 = np.linspace(-2.0, 2.0, 40)
    x2 = np.sin(np.arange(40.0))
    y = (2.0 * x1 + 0.05 * np.cos(np.arange(40.0))).reshape(-1, 1)
    x = np.column_stack([x1, x2])

    result = forward_regression(
        y, x, ["strong", "noise"], criterion="p_value", p_value_threshold=0.05
    )

    assert result.selected_features == ["strong"]


def test_forward_regression_raises_when_no_feature_improves() -> None:
    y = np.arange(20.0).reshape(-1, 1)
    x = np.column_stack([np.sin(np.arange(20.0)), np.cos(np.arange(20.0))])

    with pytest.raises(ValueError, match="No feature improved"):
        forward_regression(
            y, x, ["a", "b"], criterion="p_value", p_value_threshold=1e-12
        )


def test_step_result_repr_is_diagnostic() -> None:
    assert (
        repr(StepResult(1, "value", 0.1234567, "r2"))
        == "Step 1: +'value' (r2=0.123457)"
    )
