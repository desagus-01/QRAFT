from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.construction.optimization.moments import PolicyInputs
from qraft.core.panel import ScenarioPanel
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths


def _cash_path(tmp_path) -> str:
    path = tmp_path / "cash.csv"
    path.write_text("date,DFF\n2024-01-03,0.0\n")
    return str(path)


def _forecast_paths() -> ForecastPaths:
    return ForecastPaths(
        asset_paths={
            "A": np.array([[101.0, 102.0], [99.0, 100.0]]),
            "B": np.array([[51.0, 52.0], [49.0, 48.0]]),
        },
        dates=pl.Series(
            "date",
            [
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
            ],
        ),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 100.0, "B": 50.0},
        universe=AssetUniverse.factors_free(["A", "B"]),
    )


def _history() -> ScenarioPanel:
    return ScenarioPanel.from_prices(
        pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "A": [100.0, 102.0, 104.0],
                "B": [50.0, 51.0, 52.0],
            }
        )
    )


def test_policy_inputs_from_arrays_does_not_require_scenarios() -> None:
    inputs = PolicyInputs.from_arrays(
        assets=["A", "B"],
        mean=np.array([[0.01, 0.02], [0.005, 0.01]]),
        covariances=np.array(
            [
                [[0.04, 0.01], [0.01, 0.09]],
                [[0.05, 0.00], [0.00, 0.10]],
            ]
        ),
        cash_return=0.0,
    )

    assert inputs.n_horizons == 2
    assert inputs.n_assets == 2
    assert inputs.n_scenarios == 1
    np.testing.assert_allclose(inputs.require_mean()[0], [0.01, 0.02])

    with pytest.raises(ValueError, match="scenario_returns"):
        inputs.require_scenarios()


def test_policy_inputs_from_policy_sources_historical_mean_and_cvar_risk(
    tmp_path,
) -> None:
    inputs = PolicyInputs.from_policy_sources(
        forecasts=_forecast_paths(),
        cash_path=_cash_path(tmp_path),
        expected_returns="historical",
        history=_history(),
        risk="cvar",
        expectation_tolerance=None,
    )

    assert inputs.assets == ["A", "B"]
    assert inputs.n_horizons == 2
    assert inputs.n_assets == 2
    assert inputs.covariances is None
    assert inputs.scenario_returns is not None
    assert inputs.scenario_returns.shape == (2, 2, 2)
    assert inputs.require_mean().shape == (2, 2)


def test_policy_inputs_from_policy_sources_covariance_risk_only(tmp_path) -> None:
    inputs = PolicyInputs.from_policy_sources(
        forecasts=_forecast_paths(),
        cash_path=_cash_path(tmp_path),
        expected_returns="forecast",
        risk="covariance",
        expectation_tolerance=None,
    )

    assert inputs.scenario_returns is None
    assert inputs.scenario_probs is None
    assert inputs.covariances is not None
    assert inputs.covariances.shape == (2, 2, 2)
    inputs.require_covariances()
    with pytest.raises(ValueError, match="scenario_returns"):
        inputs.require_scenarios()


def test_policy_inputs_from_policy_sources_historical_mean_decay(tmp_path) -> None:
    inputs = PolicyInputs.from_policy_sources(
        forecasts=_forecast_paths(),
        cash_path=_cash_path(tmp_path),
        expected_returns="historical",
        history=_history(),
        risk="both",
        mean_decay=0.5,
        expectation_tolerance=None,
    )

    assert inputs.mean is not None
    assert inputs.covariances is not None
    assert inputs.mean.shape == (2, 2)
    assert inputs.covariances.shape == (2, 2, 2)
    np.testing.assert_allclose(inputs.mean[1], 0.5 * inputs.mean[0])


def test_policy_inputs_validate_required_fields() -> None:
    inputs = PolicyInputs.from_arrays(
        assets=["A"],
        covariances=np.array([[[0.01]]]),
    )

    inputs.require_covariances()
    with pytest.raises(ValueError, match="mean"):
        inputs.require_mean()
