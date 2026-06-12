import numpy as np
import polars as pl

from qraft.forecast.time_series.preprocessing import white_noise
from qraft.forecast.time_series.tests.iid import (
    PerAssetTestResult,
    _adaptive_permutation_stop,
    copula_lag_independence_test,
    independence_permutation_test,
    sw_mc,
)


def test_vectorized_sw_permutation_matches_fallback_path() -> None:
    rng = np.random.default_rng(123)
    pair = rng.uniform(size=(24, 2))
    prob = np.full(pair.shape[0], 1.0 / pair.shape[0])

    vectorized = independence_permutation_test(
        pair,
        prob,
        ("x", "x_lag"),
        iter=9,
        mc_iters=17,
        rng=np.random.default_rng(42),
    )
    fallback = independence_permutation_test(
        pair,
        prob,
        ("x", "x_lag"),
        stat_fun=lambda arr, p, points: sw_mc(arr, p, points),
        iter=9,
        mc_iters=17,
        rng=np.random.default_rng(42),
    )

    assert vectorized.stat == fallback.stat
    assert vectorized.p_val == fallback.p_val
    assert vectorized.reject_null == fallback.reject_null


def test_copula_lag_independence_accepts_iteration_controls() -> None:
    data = pl.DataFrame({"x": np.linspace(0.05, 0.95, 12)})
    prob = np.full(data.height, 1.0 / data.height)

    res = copula_lag_independence_test(
        copula=data,
        prob=prob,
        lags=1,
        assets=["x"],
        seed=7,
        mc_iters=5,
        perm_test_iters=3,
        perm_test_min_iters=2,
        perm_test_batch_iters=1,
    )

    assert set(res) == {"x"}
    assert set(res["x"].results) == {"lag_1"}


def test_copula_lag_independence_uses_explicit_seed() -> None:
    data = pl.DataFrame({"x": np.linspace(0.05, 0.95, 12)})
    prob = np.full(data.height, 1.0 / data.height)

    first = copula_lag_independence_test(
        copula=data,
        prob=prob,
        lags=1,
        assets=["x"],
        seed=1,
        mc_iters=5,
        perm_test_iters=4,
        perm_test_min_iters=4,
    )
    second = copula_lag_independence_test(
        copula=data,
        prob=prob,
        lags=1,
        assets=["x"],
        seed=1,
        mc_iters=5,
        perm_test_iters=4,
        perm_test_min_iters=4,
    )

    assert first["x"].results["lag_1"] == second["x"].results["lag_1"]


def test_run_iid_complex_passes_configured_iteration_controls(monkeypatch) -> None:
    captured: dict[str, float | int] = {}

    class DummyCopulaMarginalModel:
        copula_grades = pl.DataFrame({"x": [0.2, 0.4, 0.6, 0.8]})
        prob = np.full(4, 0.25)

        @classmethod
        def from_data_and_prob(cls, data, prob):
            return cls()

    def fake_copula_lag_independence_test(**kwargs):
        captured["mc_iters"] = kwargs["mc_iters"]
        captured["perm_test_iters"] = kwargs["perm_test_iters"]
        captured["perm_test_min_iters"] = kwargs["perm_test_min_iters"]
        captured["perm_test_batch_iters"] = kwargs["perm_test_batch_iters"]
        captured["perm_test_ci_level"] = kwargs["perm_test_ci_level"]
        captured["significance_level"] = kwargs["significance_level"]
        return {}

    monkeypatch.setattr(
        white_noise,
        "CopulaMarginalModel",
        DummyCopulaMarginalModel,
    )
    monkeypatch.setattr(
        white_noise,
        "copula_lag_independence_test",
        fake_copula_lag_independence_test,
    )

    white_noise._run_iid_complex(
        data=pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        prob=np.full(4, 0.25),
        assets=["x"],
        lags=1,
        mc_iters=13,
        perm_test_iters=11,
        perm_test_min_iters=7,
        perm_test_batch_iters=5,
        perm_test_ci_level=0.95,
        significance_level=0.1,
        seed=3,
    )

    assert captured == {
        "mc_iters": 13,
        "perm_test_iters": 11,
        "perm_test_min_iters": 7,
        "perm_test_batch_iters": 5,
        "perm_test_ci_level": 0.95,
        "significance_level": 0.1,
    }


def test_check_white_noise_uses_explicit_seed(monkeypatch) -> None:
    captured: dict[str, int | None] = {}

    def fake_run_iid_simple(data, prob, assets, lags):
        return {
            "ellipsoid": {
                asset: PerAssetTestResult(results={}, rejected=[]) for asset in assets
            },
            "ks": {
                asset: PerAssetTestResult(results={}, rejected=[]) for asset in assets
            },
        }

    def fake_run_iid_complex(**kwargs):
        captured["seed"] = kwargs["seed"]
        return {
            "copula": {
                asset: PerAssetTestResult(results={}, rejected=[])
                for asset in kwargs["assets"]
            }
        }

    monkeypatch.setattr(white_noise, "_run_iid_simple", fake_run_iid_simple)
    monkeypatch.setattr(white_noise, "_run_iid_complex", fake_run_iid_complex)

    white_noise.check_white_noise(
        data=pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        prob=np.full(4, 0.25),
        assets=["x"],
        seed=123,
    )

    assert captured["seed"] == 123


def test_adaptive_permutation_stop_waits_for_minimum_and_borderline_cases() -> None:
    assert not _adaptive_permutation_stop(
        exceedances=90,
        n_iters=90,
        min_iters=100,
        significance_level=0.05,
        ci_level=0.99,
    )
    assert _adaptive_permutation_stop(
        exceedances=90,
        n_iters=100,
        min_iters=100,
        significance_level=0.05,
        ci_level=0.99,
    )
    assert not _adaptive_permutation_stop(
        exceedances=5,
        n_iters=100,
        min_iters=100,
        significance_level=0.05,
        ci_level=0.99,
    )
