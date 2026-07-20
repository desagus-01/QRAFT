"""Allocation facade for applying construction policies to market forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from qraft.construction.inputs import (
    build_optimizer_input_table,
    forecast_run_for_source,
)
from qraft.construction.policies.policies import PolicyProtocol
from qraft.construction.policies.policy_run import PolicyRun, run_policy
from qraft.construction.state import PortfolioState
from qraft.core.market import MarketData
from qraft.core.snapshot import MarketSnapshot
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.forecast.forecaster import ForecastSpec
from qraft.forecast.run import ForecastRecipeHistory, ForecastRun, ForecastStep
from qraft.risk.risk_report import PortfolioRisk


@dataclass(frozen=True, slots=True)
class Allocation:
    """Bundle market data, a policy, forecasts, and optional portfolio state."""

    market: MarketData
    policy: PolicyProtocol
    forecasts: ForecastSpec | ForecastPaths
    state: PortfolioState | None = None
    initial_cash: float = 100.0
    step_size: int = 1

    def at(self, as_of: datetime | None = None) -> PolicyRun:
        """Return a ``PolicyRun`` for the requested as-of bar."""
        if as_of is None:
            as_of = self.market.trading_bars[-1]

        snapshot = self.market.snapshot_at(
            t=as_of,
            t_next=as_of,
            step_size=self.step_size,
        )
        forecasts = self._forecast_for(snapshot)
        state = self.state
        if state is None:
            state = PortfolioState.from_cash(
                self.initial_cash,
                list(self.market.universe.assets),
                forecasts,
            )

        table = self._input_table(snapshot, forecasts)

        return run_policy(
            self.policy,
            state,
            forecasts=forecasts,
            optimizer_inputs=table.get(snapshot.t),
            as_of=snapshot.t,
        )

    def risk(self, as_of: datetime | None = None, **kw: Any) -> PortfolioRisk:
        """Return risk for a fresh allocation run at ``as_of``.

        If you also need the allocation decision, prefer ``run = at(as_of)`` followed
        by ``run.risk(**kw)`` so the forecast built for the run is reused.
        """
        return self.at(as_of).risk(**kw)

    def _forecast_for(self, snapshot: MarketSnapshot) -> ForecastPaths:
        if isinstance(self.forecasts, ForecastPaths):
            return self.forecasts

        forecast_run = forecast_run_for_source(
            [snapshot],
            self.forecasts,
            market=self.market,
        )
        return forecast_run.forecast_at(snapshot.t)

    def _input_table(
        self,
        snapshot: MarketSnapshot,
        forecasts: ForecastPaths,
    ):
        return build_optimizer_input_table(
            [snapshot],
            ForecastRun(
                recipe_history=ForecastRecipeHistory(
                    periods=(), pipeline_config=object()
                ),
                steps=(
                    ForecastStep(
                        as_of=snapshot.t,
                        recipe_period_index=0,
                        forecast=forecasts,
                        action="applied_recipe",
                    ),
                ),
            ),
            policy=self.policy,
            market=self.market,
        )
