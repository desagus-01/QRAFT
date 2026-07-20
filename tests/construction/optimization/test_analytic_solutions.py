import numpy as np

from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.optimization.objectives.specs import (
    CovarianceRisk,
    ExpectedReturn,
    ObjectiveSpec,
    WeightedTerm,
)
from qraft.construction.optimization.optimization import MultiPeriodOptimizer


def _solve_mean_variance(
    mean: np.ndarray, covariance: np.ndarray, risk_aversion: float
) -> np.ndarray:
    inputs = OptimizerInputs.from_arrays(
        assets=[f"A{i}" for i in range(mean.size)],
        mean=mean[None, :],
        covariances=covariance[None, :, :],
    )
    optimizer = MultiPeriodOptimizer(
        objective=ObjectiveSpec(
            (
                WeightedTerm(1.0, ExpectedReturn()),
                WeightedTerm(risk_aversion, CovarianceRisk()),
            )
        ),
        horizons=1,
        n_assets=mean.size,
        n_scenarios=1,
        allow_borrow=True,
    )
    result = optimizer.solve(
        inputs,
        current_weights=np.zeros(mean.size),
        current_cash=1.0,
    )
    return result.target_weights


def test_one_asset_mpo_matches_mean_variance_solution() -> None:
    mean = np.array([0.02])
    covariance = np.array([[0.04]])
    risk_aversion = 2.0

    actual = _solve_mean_variance(mean, covariance, risk_aversion)
    expected = mean / (2 * risk_aversion * covariance.diagonal())

    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_multi_asset_mpo_matches_mean_variance_solution() -> None:
    mean = np.array([0.02, 0.03])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    risk_aversion = 1.5

    actual = _solve_mean_variance(mean, covariance, risk_aversion)
    expected = np.linalg.solve(2 * risk_aversion * covariance, mean)

    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
