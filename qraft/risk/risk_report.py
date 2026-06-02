from dataclasses import dataclass
from typing import Literal

from forecast.pipelines.forecasting import ForecastPaths
from polars import DataFrame
from risk.performance_attribution import (
    PortfolioPerformanceAttribution,
    portfolio_factor_attribution,
)
from risk.portfolio_execution import PortfolioSimulation
from risk.risk_attribution import (
    EffectiveBets,
    PortfolioRiskAttribution,
    RiskContributions,
)


@dataclass(frozen=True, slots=True)
class PortfolioRisk:
    horizon: int
    r2: float
    performance_attribution: PortfolioPerformanceAttribution
    risk_attribution: PortfolioRiskAttribution

    @classmethod
    def build(
        cls,
        portfolio_simulation: PortfolioSimulation,
        asset_forecasts: ForecastPaths,
        original_data: DataFrame,
        horizon: int,
    ):
        performance_attribution = portfolio_factor_attribution(
            portfolio_forecast=portfolio_simulation,
            factors_forecast=asset_forecasts.factor_paths,
            original_data=original_data,
            horizon=horizon,
            auto_select_factors=True,
            criterion="bic",
        )
        risk_attribution = PortfolioRiskAttribution.from_performance_attribution(
            performance_attribution
        )

        return cls(
            horizon=horizon,
            r2=performance_attribution.r2,
            performance_attribution=performance_attribution,
            risk_attribution=risk_attribution,
        )

    def var(self, alpha: float = 0.05) -> RiskContributions:
        return self.risk_attribution.var(alpha=alpha)

    def cvar(self, alpha: float = 0.05) -> RiskContributions:
        return self.risk_attribution.cvar(alpha=alpha)

    def effective_bets(
        self,
        method: Literal["approximate", "exact"] = "approximate",
        max_iter: int | None = None,
    ) -> EffectiveBets:
        return self.risk_attribution.effective_bets(method=method, max_iter=max_iter)
