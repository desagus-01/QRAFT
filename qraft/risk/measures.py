from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.lib.array_utils import normalize_axis_index
from numpy.typing import NDArray
from portfolio.forecast import PortfolioForecast
from scenarios.panel import ScenarioPanel
from scenarios.types import ProbVector


@dataclass(frozen=True, slots=True)
class LossDistribution:
    """
    Single-horizon loss distribution.

    This is a semantic risk object built on top of ScenarioPanel.

    Contract
    --------
    panel:
        ScenarioPanel containing exactly one required column: "loss".

    horizon:
        Forecast horizon represented by this loss distribution.
    """

    panel: ScenarioPanel
    horizon: int

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ValueError("horizon must be >= 0")

        if "loss" not in self.panel.values.columns:
            raise ValueError("LossDistribution.panel must contain a 'loss' column")

        if self.panel.values.height == 0:
            raise ValueError("LossDistribution cannot be empty")

        extra_cols = [c for c in self.panel.values.columns if c != "loss"]
        if extra_cols:
            raise ValueError(
                "LossDistribution should contain only the 'loss' column. "
                f"Found extra columns: {extra_cols}"
            )

    @classmethod
    def from_values(
        cls,
        losses: NDArray[np.floating],
        prob: ProbVector,
        horizon: int,
    ) -> "LossDistribution":
        losses = np.asarray(losses, dtype=float)

        if losses.ndim != 1:
            raise ValueError(
                f"losses must be 1-D for a single horizon; got shape={losses.shape}"
            )

        panel = ScenarioPanel(
            values=pl.DataFrame({"loss": losses}),
            dates=None,
            prob=prob,
        )

        return cls(panel=panel, horizon=horizon)

    @classmethod
    def from_panel(
        cls,
        panel: ScenarioPanel,
        horizon: int,
        *,
        loss_col: str = "loss",
    ) -> "LossDistribution":
        if loss_col not in panel.values.columns:
            raise KeyError(f"panel must contain loss column {loss_col!r}")

        loss_panel = ScenarioPanel(
            values=panel.values.select(loss_col).rename({loss_col: "loss"}),
            dates=None,
            prob=panel.prob,
        )

        return cls(panel=loss_panel, horizon=horizon)

    @classmethod
    def from_portfolio_forecast(
        cls,
        portfolio_forecast: PortfolioForecast,
        horizon: int,
    ) -> LossDistribution:
        horizon_panel = portfolio_forecast.at_period(horizon)
        return cls.from_panel(horizon_panel, horizon=horizon, loss_col="loss")

    @property
    def values(self) -> NDArray[np.floating]:
        return self.panel.values["loss"].to_numpy()

    @property
    def probs(self) -> ProbVector:
        return self.panel.prob

    def tail_panel(self, alpha: float = 0.05) -> ScenarioPanel:
        """
        Return the conditional tail loss distribution as a ScenarioPanel.

        The returned probabilities are renormalised within the tail.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1")

        cutoff = self.var(alpha=alpha, method="empirical")

        tail = self.panel.values.with_columns(prob=pl.Series(self.panel.prob)).filter(
            pl.col("loss") >= cutoff
        )

        if tail.height == 0:
            raise ValueError("Tail selection is empty")

        tail_prob = tail["prob"].to_numpy()
        prob_sum = float(tail_prob.sum())

        if prob_sum <= 0:
            raise ValueError("Tail probability mass is zero")

        return ScenarioPanel(
            values=tail.drop("prob"),
            dates=None,
            prob=tail_prob / prob_sum,
        )

    def var(
        self,
        *,
        alpha: float = 0.05,
        method: Literal["empirical", "quantile"] = "empirical",
    ) -> float:
        prob = self.probs if method == "empirical" else None

        return float(
            var(
                distribution=self.values,
                prob=prob,
                method=method,
                alpha=alpha,
                distribution_type="loss",
            )
        )

    def cvar(
        self,
        *,
        alpha: float = 0.05,
        method: Literal["empirical", "quantile"] = "empirical",
    ) -> float:
        if method == "empirical":
            tail = self.tail_panel(alpha=alpha)
            return float((tail.values["loss"].to_numpy() * tail.prob).sum())

        return float(
            cvar(
                distribution=self.values,
                prob=self.probs,
                method=method,
                alpha=alpha,
                distribution_type="loss",
            )
        )


def _loss_value_from_pnl(
    performance: NDArray[np.floating],
) -> NDArray[np.floating]:
    return -performance


def _tail_cutoff(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    method: Literal["empirical", "quantile"],
    alpha: float,
    axis: int,
    distribution_type: Literal["pnl", "loss"],
) -> NDArray[np.floating]:
    q = alpha if distribution_type == "pnl" else 1 - alpha

    if method == "empirical" and prob is not None:
        return np.quantile(
            distribution,
            q,
            axis=axis,
            method="inverted_cdf",
            weights=prob,
        )

    if method == "quantile" and prob is None:
        return np.quantile(distribution, q, axis=axis)

    raise ValueError("Must choose either empirical with prob or quantile without prob.")


def var(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    method: Literal["empirical", "quantile"] = "quantile",
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    axis = normalize_axis_index(axis, distribution.ndim)

    cutoff = _tail_cutoff(
        distribution=distribution,
        prob=prob,
        method=method,
        alpha=alpha,
        axis=axis,
        distribution_type=distribution_type,
    )

    return cutoff if distribution_type == "loss" else -cutoff


def cvar(
    distribution: NDArray[np.floating],
    prob: ProbVector | None,
    method: Literal["empirical", "quantile"] = "quantile",
    alpha: float = 0.05,
    axis: int = 0,
    *,
    distribution_type: Literal["pnl", "loss"] = "loss",
) -> NDArray[np.floating]:
    axis = normalize_axis_index(axis, distribution.ndim)

    cutoff = _tail_cutoff(
        distribution=distribution,
        prob=prob if method == "empirical" else None,
        method=method,
        alpha=alpha,
        axis=axis,
        distribution_type=distribution_type,
    )

    expanded_cutoff = np.expand_dims(cutoff, axis=axis)

    if distribution_type == "pnl":
        tail_mask = distribution <= expanded_cutoff
        tail_values = np.where(tail_mask, distribution, np.nan)
        return -np.nanmean(tail_values, axis=axis)

    tail_mask = distribution >= expanded_cutoff
    tail_values = np.where(tail_mask, distribution, np.nan)
    return np.nanmean(tail_values, axis=axis)
