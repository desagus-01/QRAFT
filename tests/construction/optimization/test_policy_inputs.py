from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.construction.optimization.inputs import (
    InputPlan,
    OptimizerInputs,
    RequiredOptimizerInputs,
)
from qraft.core.panel import ScenarioPanel
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths


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


def test_forecast_paths_rejects_invalid_path_probs() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ForecastPaths(
            asset_paths={"A": np.array([[101.0, 102.0], [99.0, 100.0]])},
            dates=pl.Series("date", [datetime(2024, 1, 4), datetime(2024, 1, 5)]),
            path_probs=np.array([0.5, 0.4]),
            initial_prices={"A": 100.0},
            universe=AssetUniverse.factors_free(["A"]),
        )


def test_forecast_paths_copies_and_freezes_path_probs() -> None:
    path_probs = np.array([0.5, 0.5])
    forecasts = ForecastPaths(
        asset_paths={"A": np.array([[101.0, 102.0], [99.0, 100.0]])},
        dates=pl.Series("date", [datetime(2024, 1, 4), datetime(2024, 1, 5)]),
        path_probs=path_probs,
        initial_prices={"A": 100.0},
        universe=AssetUniverse.factors_free(["A"]),
    )

    path_probs[0] = 1.0

    np.testing.assert_allclose(forecasts.path_probs, [0.5, 0.5])
    assert not forecasts.path_probs.flags.writeable


def test_forecast_paths_does_not_mutate_caller_asset_paths_dict() -> None:
    raw = [[101.0, 102.0], [99.0, 100.0]]
    asset_paths = {"A": raw}

    forecasts = ForecastPaths(
        asset_paths=asset_paths,
        dates=pl.Series("date", [datetime(2024, 1, 4), datetime(2024, 1, 5)]),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 100.0},
        universe=AssetUniverse.factors_free(["A"]),
    )

    assert asset_paths["A"] is raw
    assert isinstance(forecasts.asset_paths["A"], np.ndarray)


def test_forecast_paths_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        ForecastPaths(
            asset_paths={"A": np.array([[101.0, 0.0], [99.0, 100.0]])},
            dates=pl.Series("date", [datetime(2024, 1, 4), datetime(2024, 1, 5)]),
            path_probs=np.array([0.5, 0.5]),
            initial_prices={"A": 100.0},
            universe=AssetUniverse.factors_free(["A"]),
        )


def test_forecast_paths_rejects_non_finite_paths() -> None:
    with pytest.raises(ValueError, match="NaN or inf"):
        ForecastPaths(
            asset_paths={"A": np.array([[101.0, np.inf], [99.0, 100.0]])},
            dates=pl.Series("date", [datetime(2024, 1, 4), datetime(2024, 1, 5)]),
            path_probs=np.array([0.5, 0.5]),
            initial_prices={"A": 100.0},
            universe=AssetUniverse.factors_free(["A"]),
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


def test_optimizer_inputs_from_arrays_does_not_require_scenarios() -> None:
    inputs = OptimizerInputs.from_arrays(
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


def test_optimizer_inputs_from_policy_sources_historical_mean_and_cvar_risk() -> None:
    inputs = OptimizerInputs.from_policy_sources(
        forecasts=_forecast_paths(),
        cash_return=0.0,
        expected_returns="historical",
        history=_history(),
        required_inputs=RequiredOptimizerInputs(scenarios=True),
        expectation_tolerance=None,
    )

    assert inputs.assets == ["A", "B"]
    assert inputs.n_horizons == 2
    assert inputs.n_assets == 2
    assert inputs.covariances is None
    assert inputs.scenario_returns is not None
    assert inputs.scenario_returns.shape == (2, 2, 2)
    assert inputs.require_mean().shape == (2, 2)


def test_historical_log_price_mean_is_arithmetic_simple_return() -> None:
    history = ScenarioPanel.from_prices(
        pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "A": [100.0, 110.0, 99.0],
                "B": [50.0, 55.0, 49.5],
            }
        )
    )

    inputs = OptimizerInputs.from_policy_sources(
        forecasts=_forecast_paths(),
        cash_return=0.0,
        expected_returns="historical",
        history=history,
        required_inputs=RequiredOptimizerInputs(covariances=True),
    )

    log_returns = history.diff().drop_nulls()
    data = log_returns.values.select(["A", "B"]).to_numpy()
    expected = np.exp(data.mean(axis=0) + 0.5 * data.var(axis=0)) - 1.0
    np.testing.assert_allclose(inputs.require_mean()[0], expected)


def test_log_pnl_rejected_for_optimizer_inputs() -> None:
    with pytest.raises(ValueError, match="pnl_type='log'"):
        InputPlan(pnl_type="log")

    with pytest.raises(ValueError, match="pnl_type='log'"):
        OptimizerInputs.from_policy_sources(
            forecasts=_forecast_paths(),
            cash_return=0.0,
            pnl_type="log",
        )


def test_optimizer_inputs_from_policy_sources_covariance_risk_only() -> None:
    inputs = OptimizerInputs.from_policy_sources(
        forecasts=_forecast_paths(),
        cash_return=0.0,
        expected_returns="forecast",
        required_inputs=RequiredOptimizerInputs(covariances=True),
        expectation_tolerance=None,
    )

    assert inputs.scenario_returns is None
    assert inputs.scenario_probs is None
    assert inputs.covariances is not None
    assert inputs.covariances.shape == (2, 2, 2)
    inputs.require_covariances()


def test_optimizer_inputs_drops_degenerate_forecast_asset() -> None:
    forecasts = ForecastPaths(
        asset_paths={
            "A": np.array([[101.0, 102.0], [99.0, 100.0]]),
            "B": np.array([[50.0, 50.0], [50.0, 50.0]]),
        },
        dates=pl.Series("date", [datetime(2024, 1, 4), datetime(2024, 1, 5)]),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 100.0, "B": 50.0},
        universe=AssetUniverse.factors_free(["A", "B"]),
    )

    inputs = OptimizerInputs.from_policy_sources(
        forecasts=forecasts,
        cash_return=0.0,
        expected_returns="forecast",
        required_inputs=RequiredOptimizerInputs(covariances=True, scenarios=True),
    )

    assert inputs.assets == ["A"]
    assert [drop.asset for drop in inputs.dropped_assets] == ["B"]
    assert inputs.dropped_assets[0].reason == "degenerate_forecast"


def test_optimizer_inputs_from_policy_sources_historical_mean_decay() -> None:
    inputs = OptimizerInputs.from_policy_sources(
        forecasts=_forecast_paths(),
        cash_return=0.0,
        expected_returns="historical",
        history=_history(),
        required_inputs=RequiredOptimizerInputs(covariances=True, scenarios=True),
        mean_decay=0.5,
        expectation_tolerance=None,
    )

    assert inputs.mean is not None
    assert inputs.covariances is not None
    assert inputs.mean.shape == (2, 2)
    assert inputs.covariances.shape == (2, 2, 2)
    np.testing.assert_allclose(inputs.mean[1], 0.5 * inputs.mean[0])


def test_optimizer_inputs_validate_required_fields() -> None:
    inputs = OptimizerInputs.from_arrays(
        assets=["A"],
        covariances=np.array([[[0.01]]]),
    )

    inputs.require_covariances()
    with pytest.raises(ValueError, match="mean"):
        inputs.require_mean()


def test_optimizer_inputs_rejects_non_finite_scenario_returns() -> None:
    with pytest.raises(ValueError, match="scenario_returns contains"):
        OptimizerInputs.from_arrays(
            assets=["A"],
            mean=np.array([[0.01]]),
            scenario_returns=np.array([[[np.nan]]]),
            scenario_probs=np.array([1.0]),
        )


def test_optimizer_inputs_rejects_non_finite_cash_return() -> None:
    with pytest.raises(ValueError, match="cash_return contains"):
        OptimizerInputs.from_arrays(
            assets=["A"],
            mean=np.array([[0.01]]),
            cash_return=np.array([np.inf]),
        )
