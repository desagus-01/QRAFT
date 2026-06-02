from dataclasses import dataclass
from typing import Literal

import numpy as np
from construction.policy_projection import PolicyProjection
from forecast.pipelines.forecasting import ForecastPaths
from forecast.time_series.feature_selection import Criterion
from numpy.typing import NDArray
from polars import DataFrame
from risk.measures import cvar, var
from risk.performance_attribution import (
    portfolio_factor_attribution,
)
from risk.risk_attribution import (
    EffectiveBets,
    PortfolioRiskAttribution,
    RiskContributions,
)

RiskMetrics = Literal["var", "cvar"]


@dataclass(frozen=True, slots=True)
class PortfolioRisk:
    horizon: int
    r2: float
    policy_projection: PolicyProjection
    # performance_attribution: PortfolioPerformanceAttribution
    risk_attribution: PortfolioRiskAttribution

    @classmethod
    def build(
        cls,
        policy_projection: PolicyProjection,
        asset_forecasts: ForecastPaths,
        original_data: DataFrame,
        auto_select_factors: bool,
        criterion: Criterion,
        horizon: int,
    ):
        performance_attribution = portfolio_factor_attribution(
            policy_projection=policy_projection,
            factors_forecast=asset_forecasts.factor_paths,
            original_data=original_data,
            horizon=horizon,
            auto_select_factors=auto_select_factors,
            criterion=criterion,
        )
        risk_attribution = PortfolioRiskAttribution.from_performance_attribution(
            performance_attribution
        )

        return cls(
            horizon=horizon,
            r2=performance_attribution.r2,
            policy_projection=policy_projection,
            # performance_attribution=performance_attribution,
            risk_attribution=risk_attribution,
        )

    def risk_contribution(
        self, risk_metric: RiskMetrics, alpha: float = 0.05
    ) -> RiskContributions:
        return (
            self.risk_attribution.var(alpha=alpha)
            if risk_metric == "var"
            else self.risk_attribution.cvar(alpha=alpha)
        )

    def risk_at_horizon(
        self,
        risk_metric: RiskMetrics,
        method: Literal["empirical", "quantile"] = "empirical",
        alpha: float = 0.05,
    ) -> NDArray[np.floating]:
        losses = -self.policy_projection.performance_at_period(self.horizon)
        return (
            var(
                losses,
                prob=self.policy_projection.path_probs,
                method=method,
                alpha=alpha,
                axis=0,
                distribution_type="loss",
            )
            if risk_metric == "var"
            else cvar(
                losses,
                prob=self.policy_projection.path_probs,
                method=method,
                alpha=alpha,
                axis=0,
                distribution_type="loss",
            )
        )

    def effective_bets(
        self,
        method: Literal["approximate", "exact"] = "approximate",
        max_iter: int | None = None,
    ) -> EffectiveBets:
        return self.risk_attribution.effective_bets(method=method, max_iter=max_iter)
