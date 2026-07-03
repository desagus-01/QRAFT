from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from qraft.core.panel import ScenarioPanel
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


def test_var_contribution_averages_quantile_window() -> None:
    panel = ScenarioPanel(
        values=pl.DataFrame(
            {
                "factor": [100.0, 10.0, 12.0, 14.0],
                "loss": [0.0, 1.0, 1.01, 1.02],
            }
        ),
        dates=pl.Series([datetime(2024, 1, 1) + timedelta(days=i) for i in range(4)]),
        prob=np.full(4, 0.25),
        kind="return",
    )

    res = risk_attribution.var_contribution(
        panel=panel,
        exposures={"factor": 2.0},
        alpha=0.5,
    )

    assert res.value == pytest.approx(1.0)
    assert res.contributions["factor"] == pytest.approx(22.0)
