import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from qraft.construction.optimization.constraints import PortfolioConstraint
from qraft.construction.optimization.moments import (
    HorizonMoments,
    MomentsConfig,
    PnL_OPTIONS,
)
from qraft.construction.optimization.objectives.specs import (
    HoldingCost,
    TransactionCost,
)
from qraft.construction.optimization.optimization import (
    MPOFailure,
    MPOResult,
    MultiPeriodOptimizer,
)
from qraft.construction.optimization.presets import PreMadeObjectives
from qraft.construction.optimization.problem import MPOProblem
from qraft.construction.policies.policy_decision import PolicyDecision
from qraft.construction.state import PortfolioState, align_state_to_assets
from qraft.forecast.forecast_paths import ForecastPaths

logger = logging.getLogger(__name__)


class PolicyProtocol(Protocol):
    name: str

    def decide(
        self, state: PortfolioState, forecasts: ForecastPaths
    ) -> PolicyDecision: ...


def _decision_from_mpo(
    result: MPOResult, cash_return: NDArray[np.floating]
) -> PolicyDecision:
    return PolicyDecision(
        asset_order=result.assets,
        target_weights_risk=result.target_weights,
        target_cash_weight=result.target_cash,
        cash_return=cash_return,
        diagnostics=result,
    )


@dataclass(frozen=True, slots=True)
class EqualWeightPolicy:
    target_cash_weight: float
    name: str = "equal_weight"
    requires_forecast: ClassVar[bool] = False

    def decide(self, state: PortfolioState, forecasts: ForecastPaths) -> PolicyDecision:
        risky_weights = 1.0 - self.target_cash_weight
        n_assets = len(state.asset_order)
        target_weights = np.full(n_assets, risky_weights / n_assets)
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=target_weights,
            target_cash_weight=self.target_cash_weight,
            cash_return=np.zeros(1),
            diagnostics=None,
        )


@dataclass(frozen=True, slots=True)
class MPOPolicy:
    requires_forecast: ClassVar[bool] = True
    problem: MPOProblem
    moments_config: MomentsConfig
    name: str = "mpo"
    _optimizer_cache: dict[tuple[tuple[str, ...], int, int], MultiPeriodOptimizer] = (
        field(default_factory=dict, compare=False, repr=False)
    )

    @classmethod
    def preset(
        cls,
        objective_type: PreMadeObjectives,
        risk_aversion: float,
        cash_path: str,
        *,
        name: str = "mpo",
        horizons: int | None = None,
        pnl_type: PnL_OPTIONS = "relative",
        expectation_tolerance: float | None = 1.0,
        step_size: int = 1,
        periods_per_year: int = 252,
        alpha: float | None = 0.05,
        constraints: Sequence[PortfolioConstraint] = (),
        allow_borrow: bool = False,
        max_iter: int = 200,
        transaction_cost: TransactionCost | None = None,
        transaction_cost_weight: float = 1.0,
        holding_cost: HoldingCost | None = None,
        holding_cost_weight: float = 1.0,
        **solver_options: Any,
    ) -> "MPOPolicy":
        return cls(
            problem=MPOProblem.preset(
                objective_type,
                risk_aversion,
                alpha=alpha,
                constraints=constraints,
                allow_borrow=allow_borrow,
                max_iter=max_iter,
                transaction_cost=transaction_cost,
                transaction_cost_weight=transaction_cost_weight,
                holding_cost=holding_cost,
                holding_cost_weight=holding_cost_weight,
                **solver_options,
            ),
            moments_config=MomentsConfig(
                cash_path=cash_path,
                horizons=horizons,
                pnl_type=pnl_type,
                expectation_tolerance=expectation_tolerance,
                step_size=step_size,
                periods_per_year=periods_per_year,
            ),
            name=name,
        )

    def compute_moments(self, forecasts: ForecastPaths) -> HorizonMoments:
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

    def _get_optimizer(self, moments: HorizonMoments) -> MultiPeriodOptimizer:
        key = (tuple(moments.assets), moments.n_horizons, moments.n_scenarios)
        optimizer = self._optimizer_cache.get(key)
        if optimizer is None:
            optimizer = self.problem.compile(
                horizons=moments.n_horizons,
                n_assets=moments.n_assets,
                n_scenarios=moments.n_scenarios,
            )
            self._optimizer_cache[key] = optimizer
        return optimizer

    def decide(self, state: PortfolioState, forecasts: ForecastPaths) -> PolicyDecision:
        moments = self.compute_moments(forecasts)
        current_weights, current_cash, dropped, dropped_weight = align_state_to_assets(
            state, moments.assets
        )
        if dropped:
            logger.warning(
                "decide(): %d asset(s) dropped by compute_moments %s — "
                "their combined weight (%.4f) transferred to cash for reallocation.",
                len(dropped),
                sorted(dropped),
                dropped_weight,
            )

        optimizer = self._get_optimizer(moments)
        result = optimizer.solve_auto(
            moments=moments,
            current_weights=current_weights,
            current_cash=current_cash,
            max_iter=self.problem.max_iter,
            **self.problem.solver_options,
        )
        if isinstance(result, MPOFailure):
            raise RuntimeError(result.message)

        return _decision_from_mpo(result, cash_return=moments.cash_return)
