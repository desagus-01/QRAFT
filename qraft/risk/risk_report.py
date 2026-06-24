from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from qraft.core.metrics import cvar, var
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.risk.feature_selection import Criterion
from qraft.risk.performance_attribution import (
    portfolio_factor_attribution,
)
from qraft.risk.risk_attribution import (
    EffectiveBets,
    PortfolioRiskAttribution,
    RiskContributions,
)

if TYPE_CHECKING:
    from qraft.construction.policies import PolicyProjection

RiskMetrics = Literal["var", "cvar"]


@dataclass(frozen=True, slots=True)
class PortfolioRisk:
    horizon: int
    r2: float
    policy_projection: PolicyProjection
    risk_attribution: PortfolioRiskAttribution

    @classmethod
    def from_projection(
        cls,
        policy_projection: PolicyProjection,
        asset_forecasts: ForecastPaths,
        auto_select_factors: bool = False,
        criterion: Criterion | None = None,
    ):
        if auto_select_factors and criterion is None:
            criterion = "bic"

        horizon = asset_forecasts.n_horizons - 1
        performance_attribution = portfolio_factor_attribution(
            policy_projection=policy_projection,
            factors_forecast=asset_forecasts.factor_paths,
            initial_prices=asset_forecasts.initial_prices,
            horizon=horizon,
            date=asset_forecasts.dates[horizon],
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
        alpha: float = 0.05,
    ) -> NDArray[np.floating]:
        losses = -self.policy_projection.performance_at_period(self.horizon)
        return (
            var(
                losses,
                prob=self.policy_projection.path_probs,
                alpha=alpha,
                axis=0,
                distribution_type="loss",
            )
            if risk_metric == "var"
            else cvar(
                losses,
                prob=self.policy_projection.path_probs,
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
