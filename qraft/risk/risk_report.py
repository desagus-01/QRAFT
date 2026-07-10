from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
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
        forecasts: ForecastPaths,
        auto_select_factors: bool = False,
        criterion: Criterion | None = None,
    ):
        if auto_select_factors and criterion is None:
            criterion = "bic"

        horizon = forecasts.n_horizons - 1
        performance_attribution = portfolio_factor_attribution(
            portfolio_performance_forecast=policy_projection.performance_at_period(
                horizon
            ),
            path_probs=policy_projection.path_probs,
            factors_forecast=forecasts.factor_paths,
            initial_prices=forecasts.initial_prices,
            horizon=horizon,
            date=forecasts.dates[horizon],
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

    def summary_df(self, alpha: float = 0.05) -> pl.DataFrame:
        var_contrib = self.risk_contribution("var", alpha=alpha)
        cvar_contrib = self.risk_contribution("cvar", alpha=alpha)
        enb = self.effective_bets().effective_bets

        rows = [
            {
                "metric": "VaR",
                "value": var_contrib.value,
                "contribution": None,
                "pct_of_total": None,
            },
            {
                "metric": "CVaR",
                "value": cvar_contrib.value,
                "contribution": None,
                "pct_of_total": None,
            },
            {
                "metric": "ENB",
                "value": enb,
                "contribution": None,
                "pct_of_total": None,
            },
            {
                "metric": "r2",
                "value": self.r2,
                "contribution": None,
                "pct_of_total": None,
            },
        ]

        total = cvar_contrib.value
        for name, contribution in cvar_contrib.contributions.items():
            rows.append(
                {
                    "metric": name,
                    "value": None,
                    "contribution": contribution,
                    "pct_of_total": contribution / total if total != 0.0 else None,
                }
            )

        rows.append(
            {
                "metric": "factor_explained_fraction",
                "value": self.r2,
                "contribution": None,
                "pct_of_total": None,
            }
        )

        return pl.DataFrame(rows)

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
