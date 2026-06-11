import numpy as np

from qraft.core.estimation import effective_sample_size, weighted_ols
from qraft.forecast.time_series.tests.iid import autocorrelation_pair_test


def test_effective_sample_size_uses_weight_concentration() -> None:
    uniform = np.full(4, 0.25)
    concentrated = np.array([0.7, 0.1, 0.1, 0.1])

    assert effective_sample_size(uniform) == 4.0
    assert effective_sample_size(concentrated) < 2.0


def test_autocorrelation_pair_test_uses_effective_sample_size() -> None:
    pair = np.column_stack([np.arange(1.0, 6.0), np.arange(1.0, 6.0)])
    uniform = np.full(pair.shape[0], 1.0 / pair.shape[0])
    concentrated = np.array([0.96, 0.01, 0.01, 0.01, 0.01])

    uniform_res = autocorrelation_pair_test(pair, uniform, ("x", "x_lag"))
    concentrated_res = autocorrelation_pair_test(pair, concentrated, ("x", "x_lag"))

    assert uniform_res.stat == concentrated_res.stat
    assert concentrated_res.p_val > uniform_res.p_val


def test_weighted_ols_p_values_use_effective_sample_size() -> None:
    x = np.arange(1.0, 11.0).reshape(-1, 1)
    design = np.column_stack([np.ones(x.shape[0]), x.ravel()])
    y = (
        1.0
        + 2.0 * x.ravel()
        + np.array([0.1, -0.2, 0.05, -0.05, 0.2, -0.1, 0.15, -0.15, 0.08, -0.08])
    )
    uniform = np.full(x.shape[0], 1.0 / x.shape[0])
    concentrated = np.array([0.5, *([0.5 / 9.0] * 9)])

    uniform_fit = weighted_ols(y, design, prob=uniform)
    concentrated_fit = weighted_ols(y, design, prob=concentrated)

    assert concentrated_fit.p_values[1, 0] > uniform_fit.p_values[1, 0]
