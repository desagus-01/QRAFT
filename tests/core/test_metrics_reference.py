import numpy as np
import pytest
from scipy.stats import norm

from qraft.core.metrics import cvar, var


def test_var_and_cvar_match_discrete_atom_case() -> None:
    losses = np.array([1.0, 2.0, 3.0, 4.0])
    probabilities = np.array([0.1, 0.3, 0.4, 0.2])

    assert var(losses, probabilities, alpha=0.4) == pytest.approx(3.0)
    assert cvar(losses, probabilities, alpha=0.4) == pytest.approx(3.5)


def test_var_and_cvar_match_normal_closed_form() -> None:
    rng = np.random.default_rng(20240720)
    mean = -0.002
    sigma = 0.015
    alpha = 0.05
    losses = rng.normal(mean, sigma, 1_000_000)
    z = norm.ppf(1 - alpha)

    assert var(losses, None, alpha=alpha) == pytest.approx(mean + sigma * z, abs=1e-4)
    assert cvar(losses, None, alpha=alpha) == pytest.approx(
        mean + sigma * norm.pdf(z) / alpha, abs=1e-4
    )
