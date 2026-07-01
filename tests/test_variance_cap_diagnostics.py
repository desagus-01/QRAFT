from __future__ import annotations

import numpy as np

from qraft.forecast.simulation.engines.volatility import garch_simulation_paths
from qraft.forecast.simulation.state import _garch_unconditional_variance
from qraft.forecast.time_series.models.model_types import CompiledParams


def _params() -> CompiledParams:
    return CompiledParams(
        mu=0.0,
        ar=np.zeros(0),
        ma=np.zeros(0),
        omega=1.0,
        alpha=np.array([2.0]),
        gamma=np.zeros(0),
        beta=np.zeros(0),
    )


def test_garch_simulation_without_cap_only_applies_floor() -> None:
    innovations = np.array([[10.0, 10.0]])

    sigma2, _, diagnostics = garch_simulation_paths(
        params=_params(),
        garch_order=(1, 0, 0),
        eps_start=np.array([10.0]),
        var_start=None,
        innovations=innovations,
        var_cap=None,
        asset="A",
    )

    assert diagnostics.cap_enabled is False
    assert diagnostics.bind_count == 0
    assert sigma2[0, 0] == 201.0
    assert sigma2[0, 1] > sigma2[0, 0]


def test_garch_simulation_records_cap_binding() -> None:
    innovations = np.array([[10.0, 10.0]])

    sigma2, _, diagnostics = garch_simulation_paths(
        params=_params(),
        garch_order=(1, 0, 0),
        eps_start=np.array([10.0]),
        var_start=None,
        innovations=innovations,
        var_cap=50.0,
        unconditional_variance=25.0,
        asset="A",
    )

    assert np.all(sigma2 <= 50.0)
    assert diagnostics.cap_enabled is True
    assert diagnostics.cap_value == 50.0
    assert diagnostics.unconditional_variance == 25.0
    assert diagnostics.bind_count == 2
    assert diagnostics.step_count == 2
    assert diagnostics.bind_rate == 1.0
    assert diagnostics.bound is True
    assert diagnostics.max_raw_variance is not None
    assert diagnostics.max_raw_variance > 50.0


def test_unconditional_variance_uses_garch_lag_parameter_keys() -> None:
    variance = _garch_unconditional_variance(
        {"omega": 0.1, "alpha[1]": 0.2, "gamma[1]": 0.1, "beta[1]": 0.6},
        (1, 1, 1),
        fallback=9.0,
    )

    assert variance == 0.1 / (1.0 - 0.2 - 0.05 - 0.6)
