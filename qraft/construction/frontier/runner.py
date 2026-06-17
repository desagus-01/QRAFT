from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from qraft.construction.frontier.config import FrontierKind, MPOFrontierConfig
from qraft.construction.frontier.metrics import (
    ex_ante_metrics,
    ex_post_terminal_cvar,
    in_model_cvar,
)
from qraft.construction.frontier.result import FrontierPoint, FrontierResult
from qraft.construction.optimization.moments import HorizonMoments, MomentsConfig
from qraft.construction.optimization.optimization import (
    MPOFailure,
    MultiPeriodOptimizer,
)
from qraft.construction.optimization.problem import MPOProblem
from qraft.construction.state import PortfolioState, align_state_to_assets
from qraft.forecast.forecast_paths import ForecastPaths

logger = logging.getLogger(__name__)


class MPOFrontierRunner:
    def __init__(
        self,
        config: MPOFrontierConfig,
        moments_config: MomentsConfig,
    ) -> None:
        self.config = config
        self.moments_config = moments_config
        self._cache: dict[
            tuple[float, float, tuple[str, ...], int, int], MultiPeriodOptimizer
        ] = {}

    def _compute_moments(self, forecasts: ForecastPaths) -> HorizonMoments:
        cfg = self.moments_config
        return HorizonMoments.from_forecast_paths(
            forecast_paths=forecasts,
            cash_path=cfg.cash_path,
            horizons=cfg.horizons,
            subset=cfg.subset,
            pnl_type=cfg.pnl_type,
            expectation_tolerance=cfg.expectation_tolerance,
            step_size=cfg.step_size,
            periods_per_year=cfg.periods_per_year,
        )

    def _optimizer(
        self,
        gamma: float,
        moments: HorizonMoments,
    ) -> MultiPeriodOptimizer:
        key = (
            gamma,
            self.config.transaction_cost_weight,
            tuple(moments.assets),
            moments.n_horizons,
            moments.n_scenarios,
        )
        optimizer = self._cache.get(key)
        if optimizer is None:
            problem = MPOProblem.preset(
                objective_type=self.config.objective_type,
                risk_aversion=gamma,
                transaction_cost_weight=self.config.transaction_cost_weight,
                alpha=self.config.alpha,
                constraints=self.config.constraints,
                allow_borrow=self.config.allow_borrow,
                max_iter=self.config.max_iter,
                transaction_cost=self.config.transaction_cost,
                holding_cost=self.config.holding_cost,
                holding_cost_weight=self.config.holding_cost_weight,
            )
            optimizer = problem.compile(
                moments.n_horizons,
                moments.n_assets,
                moments.n_scenarios,
            )
            self._cache[key] = optimizer
        return optimizer

    def run_single(
        self,
        state: PortfolioState,
        forecasts: ForecastPaths,
        inputs: Mapping[str, Any] | None = None,
    ) -> FrontierResult:
        moments = self._compute_moments(forecasts)
        current_weights, current_cash, dropped, dropped_weight = align_state_to_assets(
            state, moments.assets
        )
        if dropped:
            logger.warning(
                "run_single(): %d asset(s) dropped by moment construction %s; "
                "their combined weight (%.4f) was transferred to cash.",
                len(dropped),
                sorted(dropped),
                dropped_weight,
            )

        points: list[FrontierPoint] = []
        for gamma in sorted(self.config.risk_aversions):
            result = self._optimizer(gamma, moments).solve_auto(
                moments=moments,
                current_weights=current_weights,
                current_cash=current_cash,
                inputs=dict(inputs) if inputs is not None else None,
                max_iter=self.config.max_iter,
                raise_on_failure=False,
                **self.config.solver_options,
            )
            if isinstance(result, MPOFailure):
                points.append(
                    FrontierPoint.failed(
                        gamma=gamma,
                        status=result.status,
                        message=result.message,
                    )
                )
                continue

            ante = ex_ante_metrics(result, moments)
            compute_cvar = self.config.cvar_mode == "ex_post"
            cvar_in = (
                in_model_cvar(result, moments, self.config.tail_metric_alpha)
                if compute_cvar
                else None
            )
            cvar_term = (
                ex_post_terminal_cvar(result, moments, self.config.tail_metric_alpha)
                if compute_cvar
                else None
            )

            points.append(
                FrontierPoint(
                    gamma=gamma,
                    status=result.status,
                    objective_value=result.objective_value,
                    expected_return=ante["expected_return"],
                    volatility=ante["volatility"],
                    cvar_in_model=cvar_in,
                    cvar_terminal=cvar_term,
                    turnover=result.turnover,
                    cash_weight=result.target_cash,
                    target_weights=result.target_weights_by_asset,
                    metric_kind=FrontierKind.EX_ANTE_MPO,
                )
            )

        return FrontierResult(
            risk_measure=self.config.risk_measure,
            points=points,
            kind=FrontierKind.EX_ANTE_MPO,
            assets=moments.assets,
            n_horizons=moments.n_horizons,
            n_scenarios=moments.n_scenarios,
            dropped_assets=dropped,
            dropped_weight=dropped_weight,
        )

    def run_repeated(
        self,
        dated_states: Iterable[
            tuple[PortfolioState, ForecastPaths]
            | tuple[PortfolioState, ForecastPaths, Mapping[str, Any] | None]
        ],
    ) -> list[FrontierResult]:
        results: list[FrontierResult] = []
        for item in dated_states:
            if len(item) == 2:
                state, forecasts = item
                inputs = None
            else:
                state, forecasts, inputs = item
            results.append(self.run_single(state, forecasts, inputs=inputs))
        return results
