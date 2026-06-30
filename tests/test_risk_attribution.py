import numpy as np
import pytest

from qraft.risk import risk_attribution


def test_effective_bets_rejects_non_positive_contributions(monkeypatch) -> None:
    monkeypatch.setattr(
        risk_attribution,
        "minimum_torsion_matrix",
        lambda factor_joint_distribution, prob, method, max_iter: np.eye(2),
    )
    monkeypatch.setattr(
        risk_attribution,
        "weighted_covariance",
        lambda data, prob: np.eye(2),
    )
    monkeypatch.setattr(
        risk_attribution,
        "_min_torso_factor_exposures",
        lambda min_torso_matrix, factor_exposures: np.array([1.0, -1.0]),
    )

    with pytest.raises(ValueError, match="strictly positive factor risk contributions"):
        risk_attribution.effective_bets(
            factor_joint_distribution=np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0]]),
            prob=np.array([1 / 3, 1 / 3, 1 / 3]),
            factor_exposures={"A": 1.0, "B": 1.0},
        )
