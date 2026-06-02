from dataclasses import dataclass

import numpy as np
from forecast.scenarios.panel import ScenarioPanel
from forecast.scenarios.types import ProbVector
from forecast.time_series.estimation import (
    EquationTypes,
)
from forecast.time_series.feature_selection import (
    Criterion,
)
from numpy.typing import NDArray
from polars import DataFrame
from risk.factor_ols import (
    extract_ols_components,
    factor_ols_regression,
    factors_n_horizon_performance,
)
from risk.portfolio_execution import PortfolioExecution


@dataclass(frozen=True, slots=True)
class PortfolioPerformanceAttribution:
    horizon: int
    portfolio_performance_forecast: NDArray[np.floating]
    factor_performance_forecast: dict[str, NDArray[np.floating]]
    exposures: NDArray[np.floating]
    shift_term: float
    residuals: NDArray[np.floating]
    path_probs: ProbVector
    r2: float

    @property
    def factor_names(self) -> list[str]:
        return list(self.factor_performance_forecast.keys())

    @property
    def full_exposures(self) -> dict[str, float]:
        exposures_dict: dict[str, float] = {
            name: float(self.exposures[i]) for i, name in enumerate(self.factor_names)
        }
        exposures_dict["z0"] = 1.0
        return exposures_dict

    @property
    def joint_distribution(self) -> DataFrame:
        return self.joint_panel.values

    @property
    def joint_panel(self) -> ScenarioPanel:
        values = DataFrame(self.factor_performance_forecast).with_columns(
            z0=self.residuals + self.shift_term,
            portfolio_performance=self.portfolio_performance_forecast,
        )

        return ScenarioPanel(
            values=values,
            dates=None,
            prob=self.path_probs,
        )


def portfolio_factor_attribution(
    portfolio_forecast: PortfolioExecution,
    factors_forecast: dict[str, NDArray[np.floating]],
    original_data: DataFrame,
    horizon: int,
    eq_type: EquationTypes = "c",
    is_log_price: bool = True,
    auto_select_factors: bool = False,
    criterion: Criterion | None = None,
) -> PortfolioPerformanceAttribution:
    factor_names = list(factors_forecast.keys())

    factors_cum = factors_n_horizon_performance(
        factors_forecast=factors_forecast,
        original_data=original_data,
        factors_names=factor_names,
        end_horizon=horizon,
        is_log_price=is_log_price,
    )

    portfolio_cum = portfolio_forecast.performance_at_period(period=horizon)

    factor_result = factor_ols_regression(
        factors_cum_forecast=factors_cum,
        portfolio_cum_forecast=portfolio_cum,
        factor_names=factor_names,
        auto_select_factors=auto_select_factors,
        criterion=criterion,
        prob=portfolio_forecast.path_probs,
        eq_type=eq_type,
    )

    selected = factor_result.selected_factors
    ols = factor_result.ols

    shift_term, exposures = extract_ols_components(
        ols_results=ols,
        selected_factors=selected,
    )
    return PortfolioPerformanceAttribution(
        horizon=horizon,
        portfolio_performance_forecast=portfolio_cum,
        factor_performance_forecast={k: factors_cum[k] for k in selected},
        exposures=exposures,
        shift_term=shift_term,
        residuals=ols.residuals.flatten(),
        path_probs=portfolio_forecast.path_probs,
        r2=ols.r_squared,
    )
