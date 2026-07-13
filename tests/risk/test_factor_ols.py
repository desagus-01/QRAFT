import numpy as np
import pytest

from qraft.core.estimation import OLSResults
from qraft.risk.factor_ols import (
    extract_factor_attribution_model,
    factor_cumulative_returns,
    factor_ols_regression,
)


def test_factor_cumulative_returns_uses_initial_price_denominator() -> None:
    forecasts = {
        "A": np.array([[101.0, 102.0], [103.0, 104.0]]),
        "B": np.array([[48.0, 54.0], [52.0, 56.0]]),
    }

    out = factor_cumulative_returns(forecasts, {"A": 100.0, "B": 50.0}, ["A", "B"], 1)

    assert np.allclose(out["A"], np.array([0.02, 0.04]))
    assert np.allclose(out["B"], np.array([0.08, 0.12]))


def test_factor_cumulative_returns_rejects_invalid_horizon() -> None:
    forecasts = {"A": np.array([[101.0]])}

    with pytest.raises(ValueError, match="non-negative"):
        factor_cumulative_returns(forecasts, {"A": 100.0}, ["A"], -1)
    with pytest.raises(ValueError, match="out of range"):
        factor_cumulative_returns(forecasts, {"A": 100.0}, ["A"], 1)


def test_factor_ols_regression_recovers_known_exposures() -> None:
    x1 = np.linspace(-1.0, 1.0, 20)
    x2 = np.linspace(1.0, -1.0, 20) ** 2
    y = 0.25 + 2.0 * x1 - 0.5 * x2

    result = factor_ols_regression(
        {"value": x1, "quality": x2},
        y,
        ["value", "quality"],
        eq_type="c",
    )

    assert result.selected_factors == ["value", "quality"]
    assert result.selection_result is None
    assert result.ols.feature_names_order == ["const", "value", "quality"]
    assert result.ols.res.ravel() == pytest.approx([0.25, 2.0, -0.5])
    assert result.ols.r_squared == pytest.approx(1.0)


def test_factor_ols_auto_selection_requires_criterion() -> None:
    with pytest.raises(ValueError, match="select a criterion"):
        factor_ols_regression(
            {"value": np.arange(5.0)},
            np.arange(5.0),
            ["value"],
            auto_select_factors=True,
        )


def test_extract_factor_attribution_model_maps_coefficients_by_name() -> None:
    ols = OLSResults(
        feature_names_order=["const", "A", "B"],
        res=np.array([[0.1], [2.0], [-1.0]]),
        std_errors=np.ones((3, 1)),
        t_stats=np.ones((3, 1)),
        p_values=np.ones((3, 1)),
        residuals=np.array([[0.01], [-0.02]]),
        r_squared=0.9,
        aic=1.0,
        bic=2.0,
    )

    model = extract_factor_attribution_model(ols, ["B"])

    assert model.exposures == {"B": -1.0}
    assert model.shift_term == 0.1
    assert np.array_equal(model.residuals, np.array([0.01, -0.02]))
    assert model.r2 == 0.9


def test_extract_factor_attribution_model_requires_feature_names() -> None:
    ols = OLSResults(
        feature_names_order=None,
        res=np.array([[1.0]]),
        std_errors=np.ones((1, 1)),
        t_stats=np.ones((1, 1)),
        p_values=np.ones((1, 1)),
        residuals=np.zeros((3, 1)),
        r_squared=0.0,
        aic=1.0,
        bic=2.0,
    )

    with pytest.raises(ValueError, match="feature_names_order"):
        extract_factor_attribution_model(ols, [])
