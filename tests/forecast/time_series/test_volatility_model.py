import numpy as np

from qraft.core.configs import VolatilityModelConfig
from qraft.forecast.time_series.models import volatility


class _FakeParams:
    def __init__(self, params: dict[str, float]) -> None:
        self._params = params

    def to_dict(self) -> dict[str, float]:
        return dict(self._params)

    def __len__(self) -> int:
        return len(self._params)


class _FakeArchResult:
    convergence_flag = 0
    bic = 1.0

    def __init__(self, params: dict[str, float]) -> None:
        self.params = _FakeParams(params)
        self.pvalues = np.ones(len(params))
        self.std_resid = np.array([0.1, -0.2, 0.3])
        self.resid = np.array([1.0, -2.0, 3.0])
        self.conditional_volatility = np.array([4.0, 5.0, 6.0])


def test_auto_garch_fits_on_percent_scale_and_returns_original_scale(monkeypatch):
    captured_y = []

    def fake_arch_model(y, **_kwargs):
        captured_y.append(np.asarray(y, dtype=float))

        class _Model:
            def fit(self, disp):
                assert disp == "off"
                return _FakeArchResult(
                    {"omega": 2.0, "alpha[1]": 0.1, "beta[1]": 0.8, "nu": 8.0}
                )

        return _Model()

    monkeypatch.setattr(volatility, "arch_model", fake_arch_model)

    asset_array = np.array([0.01, -0.02, 0.03])
    cfg = VolatilityModelConfig(max_p_order=1, max_q_order=1, max_o_order=0)

    res = volatility.auto_garch(asset_array, cfg=cfg)[0]

    np.testing.assert_allclose(captured_y[0], asset_array * 100.0)
    assert res.params["omega"] == 2.0 / 100.0**2
    assert res.params["alpha[1]"] == 0.1
    assert res.params["beta[1]"] == 0.8
    np.testing.assert_allclose(res.residuals, np.array([0.01, -0.02, 0.03]))
    np.testing.assert_allclose(res.conditional_volatility, np.array([0.04, 0.05, 0.06]))
    np.testing.assert_allclose(res.invariants, np.array([0.1, -0.2, 0.3]))


def test_auto_garch_marks_high_persistence_baseline_inadmissible(monkeypatch):
    def fake_arch_model(y, **_kwargs):
        assert y is not None

        class _Model:
            def fit(self, disp):
                assert disp == "off"
                return _FakeArchResult(
                    {"omega": 1.0, "alpha[1]": 0.7, "beta[1]": 0.4, "nu": 8.0}
                )

        return _Model()

    monkeypatch.setattr(volatility, "arch_model", fake_arch_model)

    cfg = VolatilityModelConfig(max_p_order=1, max_q_order=1, max_o_order=0)
    res = volatility.auto_garch(np.array([0.01, -0.02, 0.03]), cfg=cfg)[0]

    assert res.admissible is False
    assert res.persistence == 1.1
    assert res.fallback_reason == "baseline_failed_admissibility_checks"


def test_volatility_config_enables_variance_cap_by_default() -> None:
    assert VolatilityModelConfig().variance_overflow_cap_multiple == 100.0
