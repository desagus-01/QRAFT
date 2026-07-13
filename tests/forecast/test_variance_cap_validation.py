import numpy as np
import pytest

from qraft.forecast.simulation.variance_cap_validation import (
    compare_capped_vs_uncapped_simulation_risk,
)


class _State:
    def __init__(self, var_cap):
        self.var_cap = var_cap


class _Forecast:
    def __init__(self, var_cap, diagnostics):
        self.state0 = _State(var_cap)
        self.variance_cap_diagnostics = diagnostics


class _Fit:
    assets = ["A"]

    def __init__(self):
        self.simulation_forecasts = {"A": _Forecast(0.01, {"bind_rate": 0.5})}

    def simulate(self, innovation_paths):
        if self.simulation_forecasts["A"].state0.var_cap is None:
            return {"A": np.array([0.0, -0.2, 0.2])}
        return {"A": np.array([0.0, -0.1, 0.1])}


def test_compare_capped_vs_uncapped_simulation_risk_reports_deltas_and_restores_caps() -> (
    None
):
    fit = _Fit()

    result = compare_capped_vs_uncapped_simulation_risk(
        fit, np.ones((1, 3)), periods_per_year=252
    )

    assert fit.simulation_forecasts["A"].state0.var_cap == 0.01
    assert len(result) == 1
    assert result[0].asset == "A"
    assert result[0].bind_rate == 0.5
    assert result[0].annualized_vol_delta > 0.0
    assert result[0].cvar_5_delta != 0.0


class _FailingFit(_Fit):
    def simulate(self, innovation_paths):
        if self.simulation_forecasts["A"].state0.var_cap is None:
            raise RuntimeError("uncapped failed")
        return {"A": np.array([0.0, -0.1, 0.1])}


def test_compare_capped_vs_uncapped_simulation_risk_restores_caps_on_failure() -> None:
    fit = _FailingFit()

    with pytest.raises(RuntimeError, match="uncapped failed"):
        compare_capped_vs_uncapped_simulation_risk(
            fit, np.ones((1, 3)), periods_per_year=252
        )

    assert fit.simulation_forecasts["A"].state0.var_cap == 0.01
