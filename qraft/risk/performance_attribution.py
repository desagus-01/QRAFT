"""Portfolio performance attribution to forecast factor returns."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame

from qraft.core.estimation import (
    EquationTypes,
)
from qraft.core.panel import DatetimeSeries, ScenarioPanel
from qraft.core.probability.prob_vector import ProbVector
from qraft.risk.factor_ols import (
    FactorAttributionModel,
    extract_factor_attribution_model,
    factor_cumulative_returns,
    factor_ols_regression,
)
from qraft.risk.feature_selection import (
    Criterion,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioPerformanceAttribution:
    """OLS factor model and joint scenario panel for portfolio performance."""

    horizon: int
    model: FactorAttributionModel
    factor_performance_forecast: dict[str, NDArray[np.floating]]
    portfolio_performance_forecast: NDArray[np.floating]
    path_probs: ProbVector
    dates: DatetimeSeries

    @property
    def exposures(self) -> dict[str, float]:
        return self.model.exposures

    @property
    def r2(self) -> float:
        return self.model.r2

    @property
    def factor_names(self) -> list[str]:
        return list(self.model.exposures.keys())

    @property
    def full_exposures(self) -> dict[str, float]:
        return {**self.model.exposures, "idiosyncratic": 1.0}

    @property
    def joint_distribution(self) -> DataFrame:
        return self.joint_panel.values

    @cached_property
    def joint_panel(self) -> ScenarioPanel:
        values = DataFrame(self.factor_performance_forecast).with_columns(
            idiosyncratic=self.model.residuals + self.model.shift_term,
            portfolio_performance=self.portfolio_performance_forecast,
        )

        return ScenarioPanel(
            values=values,
            dates=self.dates,
            prob=self.path_probs,
        )


def portfolio_factor_attribution(
    portfolio_performance_forecast: NDArray[np.floating],
    path_probs: ProbVector,
    factors_forecast: dict[str, NDArray[np.floating]],
    initial_prices: dict[str, float],
    horizon: int,
    date,
    eq_type: EquationTypes = "c",
    auto_select_factors: bool = False,
    criterion: Criterion | None = None,
) -> PortfolioPerformanceAttribution:
    """Return factor attribution for forecast portfolio cumulative returns."""
    factor_names = list(factors_forecast.keys())

    factors_cum = factor_cumulative_returns(
        factors_forecast=factors_forecast,
        initial_prices=initial_prices,
        factors_names=factor_names,
        end_horizon=horizon,
    )

    factor_result = factor_ols_regression(
        factors_cum_forecast=factors_cum,
        portfolio_cum_forecast=portfolio_performance_forecast,
        factor_names=factor_names,
        auto_select_factors=auto_select_factors,
        criterion=criterion,
        prob=path_probs,
        eq_type=eq_type,
    )

    selected = factor_result.selected_factors
    ols = factor_result.ols

    model = extract_factor_attribution_model(
        ols_results=ols,
        selected_factors=selected,
    )
    logger.info(
        "Factor attribution: horizon=%d r2=%.4f selected_factors=%s exposures=%s",
        horizon,
        model.r2,
        selected,
        {k: f"{v:.4f}" for k, v in model.exposures.items()},
    )
    return PortfolioPerformanceAttribution(
        horizon=horizon,
        model=model,
        factor_performance_forecast={k: factors_cum[k] for k in selected},
        portfolio_performance_forecast=portfolio_performance_forecast,
        path_probs=path_probs,
        dates=pl.Series("date", [date] * len(path_probs)),
    )
