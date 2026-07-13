import numpy as np
import pytest

from qraft.forecast.simulation.simulate_paths import simulate_asset_paths
from qraft.forecast.simulation.state import SimulationForecast, UnivariateState
from qraft.forecast.time_series.models.model_types import UnivariateModel


def test_simulate_asset_paths_without_volatility_scales_innovations() -> None:
    forecast = SimulationForecast(
        model=UnivariateModel(
            mean_kind="none",
            mean_params={},
            vol_params={},
            innovation_scale=2.0,
        ),
        state0=UnivariateState(series_hist=np.array([1.0])),
        variance_cap_diagnostics={},
    )

    paths = simulate_asset_paths(forecast, np.array([1.0, -2.0]))

    assert paths.shape == (1, 2)
    assert np.array_equal(paths, np.array([[2.0, -4.0]]))


def test_simulate_asset_paths_garch_records_variance_cap_diagnostics() -> None:
    forecast = SimulationForecast(
        model=UnivariateModel(
            mean_kind="none",
            mean_params={},
            vol_params={"omega": 1.0, "alpha[1]": 2.0},
            vol_kind="garch",
            vol_order=(1, 0, 0),
        ),
        state0=UnivariateState(
            series_hist=np.array([0.0]),
            vol_residual_lags=np.array([10.0]),
            var_cap=50.0,
            unconditional_variance=25.0,
        ),
        variance_cap_diagnostics={},
    )

    paths = simulate_asset_paths(forecast, np.array([[10.0, 10.0]]), asset="A")

    assert paths.shape == (1, 2)
    assert forecast.variance_cap_diagnostics["cap_enabled"] is True
    assert forecast.variance_cap_diagnostics["bind_rate"] == 1.0


def test_simulate_asset_paths_rejects_unknown_vol_kind() -> None:
    forecast = SimulationForecast(
        model=UnivariateModel(
            mean_kind="none",
            mean_params={},
            vol_params={},
            vol_kind="bad",  # type: ignore[arg-type]
        ),
        state0=UnivariateState(series_hist=np.array([1.0])),
        variance_cap_diagnostics={},
    )

    with pytest.raises(ValueError, match="bad"):
        simulate_asset_paths(forecast, np.array([1.0]))
