from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

from qraft.core.panel import ScenarioPanel
from qraft.risk import risk_attribution
from qraft.risk.risk_attribution import PortfolioRiskAttribution
from qraft.risk.risk_report import PortfolioRisk


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


def _risk() -> PortfolioRisk:
    panel = ScenarioPanel(
        values=pl.DataFrame(
            {
                "factor_a": [3.0, 2.0, 1.0, 0.5],
                "factor_b": [1.0, 2.0, 0.5, 1.5],
                "idiosyncratic": [0.5, 0.2, 0.1, 0.3],
                "loss": [4.5, 4.2, 1.6, 2.3],
            }
        ),
        dates=pl.Series([datetime(2024, 1, 1) + timedelta(days=i) for i in range(4)]),
        prob=np.full(4, 0.25),
        kind="return",
    )
    attribution = PortfolioRiskAttribution(
        horizon=0,
        exposures={"factor_a": 1.0, "factor_b": 1.0, "idiosyncratic": 1.0},
        joint_panel=panel,
    )
    return PortfolioRisk(
        horizon=0,
        r2=0.11,
        policy_projection=None,
        risk_attribution=attribution,
    )


def test_portfolio_risk_summary_includes_idiosyncratic_and_sums_total() -> None:
    risk = _risk()

    summary = risk.summary_df(alpha=0.5)

    assert "idiosyncratic" in summary["metric"].to_list()
    assert "factor_explained_fraction" in summary["metric"].to_list()

    cvar_total = summary.filter(pl.col("metric") == "CVaR")["value"].item()
    contribution_sum = summary.filter(pl.col("contribution").is_not_null())[
        "contribution"
    ].sum()
    assert contribution_sum == pytest.approx(cvar_total)


def test_risk_contributions_plot_returns_figure() -> None:
    fig = _risk().risk_contribution("cvar", alpha=0.5).plot()

    assert isinstance(fig, Figure)
