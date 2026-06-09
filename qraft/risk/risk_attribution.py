from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from qraft.core.estimation import weighted_covariance
from qraft.core.panel import ScenarioPanel
from qraft.core.probability.prob_vector import ProbVector
from numpy.typing import NDArray
from polars import DataFrame
from qraft.risk.dimensionality_reduction import minimum_torsion_matrix
from qraft.risk.measures import var
from qraft.risk.performance_attribution import PortfolioPerformanceAttribution
from qraft.utils.visuals import plot_effective_bets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskContributions:
    risk_measure: str
    value: float
    contributions: dict[str, float]


@dataclass(frozen=True)
class EffectiveBets:
    factor_contributions: dict[str, float]
    effective_bets: float

    def plot(self) -> None:
        return plot_effective_bets(self)


@dataclass(frozen=True, slots=True)
class PortfolioRiskAttribution:
    horizon: int
    exposures: dict[str, float]
    joint_panel: ScenarioPanel

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ValueError("horizon must be >= 0")

        if "loss" not in self.joint_panel.values.columns:
            raise ValueError("joint_panel must contain a 'loss' column")

        driver_cols = risk_driver_cols(self.joint_panel)

        missing_exposures = [c for c in driver_cols if c not in self.exposures]
        if missing_exposures:
            raise KeyError(
                f"Missing exposures for risk driver columns: {missing_exposures}"
            )

        extra_exposures = [
            c for c in self.exposures if c not in self.joint_panel.values.columns
        ]
        if extra_exposures:
            raise KeyError(
                f"Exposure keys missing from joint_panel columns: {extra_exposures}"
            )

        non_finite = [k for k, v in self.exposures.items() if not np.isfinite(v)]
        if non_finite:
            raise ValueError(f"Exposures must be finite. Bad keys: {non_finite}")

    @property
    def joint_distribution(self) -> DataFrame:
        return self.joint_panel.values

    @property
    def probs(self) -> ProbVector:
        return self.joint_panel.prob

    @classmethod
    def from_performance_attribution(
        cls,
        performance_attribution: PortfolioPerformanceAttribution,
    ) -> PortfolioRiskAttribution:
        perf_panel = performance_attribution.joint_panel

        risk_values = perf_panel.values.with_columns(
            loss=-pl.col("portfolio_performance")
        ).drop("portfolio_performance")

        return cls(
            horizon=performance_attribution.horizon,
            exposures={
                factor: -exposure
                for factor, exposure in performance_attribution.full_exposures.items()
            },
            joint_panel=ScenarioPanel(
                values=risk_values,
                dates=None,
                prob=perf_panel.prob,
            ),
        )

    def var(self, alpha: float = 0.05) -> RiskContributions:
        return var_contribution(
            panel=self.joint_panel,
            exposures=self.exposures,
            alpha=alpha,
        )

    def cvar(self, alpha: float = 0.05) -> RiskContributions:
        return cvar_contribution(
            panel=self.joint_panel,
            exposures=self.exposures,
            alpha=alpha,
        )

    def effective_bets(
        self,
        method: Literal["approximate", "exact"] = "approximate",
        max_iter: int | None = None,
    ) -> EffectiveBets:
        driver_cols = risk_driver_cols(self.joint_panel)
        factors = self.joint_panel.values.select(driver_cols).to_numpy()

        return effective_bets(
            factor_joint_distribution=factors,
            factor_exposures={k: self.exposures[k] for k in driver_cols},
            prob=self.joint_panel.prob,
            method=method,
            max_iter=max_iter,
        )


def risk_driver_cols(panel: ScenarioPanel) -> list[str]:
    if "loss" not in panel.values.columns:
        raise ValueError("panel must contain a 'loss' column")

    return [c for c in panel.values.columns if c != "loss"]


def with_prob(panel: ScenarioPanel) -> DataFrame:
    return panel.values.with_columns(prob=pl.Series(panel.prob))


def validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")


def validate_exposures_for_panel(
    panel: ScenarioPanel,
    exposures: dict[str, float],
) -> list[str]:
    """
    Validate exposures and return ordered risk-driver columns.
    """
    driver_cols = risk_driver_cols(panel)

    missing = [c for c in driver_cols if c not in exposures]
    if missing:
        raise KeyError(f"Missing exposures for risk drivers: {missing}")

    extra = [c for c in exposures if c not in panel.values.columns]
    if extra:
        raise KeyError(f"Exposure keys missing from panel columns: {extra}")

    return driver_cols


def get_var_data(
    panel: ScenarioPanel,
    alpha: float,
) -> DataFrame:
    """
    Return all scenarios in the VaR/CVaR tail.
    """
    validate_alpha(alpha)

    panel_with_prob = with_prob(panel)
    loss_array = panel_with_prob["loss"].to_numpy()
    prob_array = panel_with_prob["prob"].to_numpy()
    cutoff = float(
        var(
            loss_array,
            prob=prob_array,
            method="empirical",
            alpha=alpha,
            distribution_type="loss",
        )
    )

    return panel_with_prob.filter(pl.col("loss") >= cutoff).sort("loss")


def get_var_row(
    panel: ScenarioPanel,
    alpha: float,
) -> DataFrame:
    """
    Return the first row crossing the VaR quantile threshold.
    """
    var_data = get_var_data(panel, alpha).head(1)

    if var_data.height == 0:
        raise ValueError("VaR row is empty")

    return var_data


def cvar_contribution(
    panel: ScenarioPanel,
    exposures: dict[str, float],
    alpha: float = 0.05,
) -> RiskContributions:
    """
    Compute CVaR contribution decomposition from a risk attribution panel.
    """
    driver_cols = validate_exposures_for_panel(panel, exposures)
    cvar_tail = get_var_data(panel, alpha)

    tail_prob_sum = float(cvar_tail["prob"].sum())
    if tail_prob_sum <= 0:
        raise ValueError("Tail probability mass is zero")

    weighted_driver_means = {
        c: float(
            cvar_tail.select(
                (pl.col(c) * pl.col("prob")).sum() / pl.col("prob").sum()
            ).item()
        )
        for c in driver_cols
    }

    loss_cvar = float(
        cvar_tail.select(
            (pl.col("loss") * pl.col("prob")).sum() / pl.col("prob").sum()
        ).item()
    )

    contributions = {c: weighted_driver_means[c] * exposures[c] for c in driver_cols}

    return RiskContributions(
        risk_measure="cvar",
        value=loss_cvar,
        contributions=contributions,
    )


def var_contribution(
    panel: ScenarioPanel,
    exposures: dict[str, float],
    alpha: float = 0.05,
) -> RiskContributions:
    """
    Compute VaR contribution decomposition from a risk attribution panel.

    Parameters
    ----------
    panel
        ScenarioPanel containing risk-driver columns plus a required "loss" column.

    exposures
        Loss-space exposures aligned with the risk-driver columns.

    alpha
        Tail probability. alpha=0.05 computes the 95% loss-tail VaR.
    """
    driver_cols = validate_exposures_for_panel(panel, exposures)
    var_row = get_var_row(panel, alpha)

    contributions = {
        c: float(var_row.select(c).item()) * float(exposures[c]) for c in driver_cols
    }

    return RiskContributions(
        risk_measure="var",
        value=float(var_row.select("loss").item()),
        contributions=contributions,
    )


def _min_torso_factor_exposures(
    min_torso_matrix: NDArray[np.floating],
    factor_exposures: dict[str, float],
) -> NDArray[np.floating]:
    inv_min_torso = np.linalg.inv(min_torso_matrix)
    exposures = np.array(
        [v for v in factor_exposures.values() if isinstance(v, (float, np.floating))],
        dtype=float,
    )
    return inv_min_torso.T @ exposures


def effective_bets(
    factor_joint_distribution: NDArray[np.floating],
    factor_exposures: dict[str, float],
    prob: ProbVector,
    method: Literal["approximate", "exact"] = "approximate",
    max_iter: int | None = None,
) -> EffectiveBets:
    """
    Compute effective bets and minimum-torsion factor risk contributions.
    """

    if factor_joint_distribution.ndim != 2:
        raise ValueError(
            "factor_joint_distribution must be 2-D "
            f"(n_scenarios, n_factors); got shape={factor_joint_distribution.shape}"
        )

    factor_keys = list(factor_exposures.keys())

    if factor_joint_distribution.shape[1] != len(factor_keys):
        raise ValueError(
            "Number of factor columns does not match number of exposures. "
            f"Got {factor_joint_distribution.shape[1]} columns and "
            f"{len(factor_keys)} exposures."
        )

    min_torso_matrix = minimum_torsion_matrix(
        factor_joint_distribution,
        prob,
        method,
        max_iter,
    )

    exposures = np.array([factor_exposures[k] for k in factor_keys], dtype=float)
    covariance = weighted_covariance(data=factor_joint_distribution, prob=prob)

    min_torso_exposures = _min_torso_factor_exposures(
        min_torso_matrix,
        factor_exposures,
    )

    transformed_covariance_exposure = min_torso_matrix @ covariance @ exposures
    portfolio_variance = float(exposures @ covariance @ exposures)

    if np.isclose(portfolio_variance, 0.0):
        raise ValueError("Portfolio variance is zero; cannot compute contributions")

    factor_risk_contribution = (
        min_torso_exposures * transformed_covariance_exposure / portfolio_variance
    )

    enb = float(
        np.exp(-np.sum(factor_risk_contribution * np.log(factor_risk_contribution)))
    )

    factor_contributions = {
        k: float(v) for k, v in zip(factor_keys, factor_risk_contribution)
    }

    return EffectiveBets(
        factor_contributions=factor_contributions,
        effective_bets=enb,
    )
