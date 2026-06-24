import numpy as np

from qraft.core.probability.distributions import uniform_probs
from qraft.forecast.time_series.transforms.detrend import polynomial_detrend
from qraft.forecast.time_series.transforms.inverses import PolynomialInverseSpec


def test_polynomial_detrend_uses_scaled_time_basis():
    n_obs = 2500
    x = np.arange(n_obs, dtype=float) / (n_obs - 1)
    series = 2.0 * x**3 - 0.5 * x + 4.0

    residuals, betas = polynomial_detrend(
        series,
        prob=uniform_probs(n_obs),
        polynomial_order=3,
    )

    np.testing.assert_allclose(residuals.ravel(), 0.0, atol=1e-10)
    np.testing.assert_allclose(
        betas.ravel(), np.array([2.0, 0.0, -0.5, 4.0]), atol=1e-9
    )


def test_polynomial_inverse_evaluates_future_scaled_time():
    n_obs = 2500
    betas = np.array([2.0, 0.0, -0.5, 4.0])
    spec = PolynomialInverseSpec(order=3, betas=betas, n_obs=n_obs)

    restored = spec.inverse_for_forecasts(np.zeros((1, 2)), start_x=n_obs)

    future_x = np.array([n_obs, n_obs + 1], dtype=float) / (n_obs - 1)
    expected = np.polyval(betas, future_x)
    np.testing.assert_allclose(restored, expected.reshape(1, -1))
