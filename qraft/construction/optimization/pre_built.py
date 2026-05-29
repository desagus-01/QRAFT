from typing import Any

import numpy as np
from construction.optimization.constraints import PortfolioConstraint
from construction.optimization.moments import HorizonMoments
from construction.optimization.objectives.specs import CVaRCuttingPlane
from construction.optimization.optimization import (
    MPOResult,
    MultiPeriodOptimizer,
    PreMadeObjectives,
)
from numpy.typing import NDArray


def multi_period_optimization(
    objective_type: PreMadeObjectives,
    step: int,
    n_assets: int,
    risk_aversion: float,
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    current_cash: float | None = None,
    cvar_alpha: float | None = 0.05,
    constraints: list[PortfolioConstraint] | None = None,
    inputs: dict[str, Any] | None = None,
    allow_borrow: bool = False,
    max_iter: int = 200,
    **solver_options,
) -> MPOResult:
    """
    Convenience wrapper around :class:`~portfolio.policy.optimization.MultiPeriodOptimizer`.
    """
    optimizer = MultiPeriodOptimizer.from_pre_built(
        objective_type=objective_type,
        horizons=step,
        n_assets=n_assets,
        n_scenarios=moments.scenario_returns.shape[0],
        risk_aversion=risk_aversion,
        cvar_alpha=cvar_alpha,
        constraints=constraints,
        allow_borrow=allow_borrow,
    )

    uses_cutting_plane = any(
        isinstance(term.spec, CVaRCuttingPlane) for term in optimizer.objective.terms
    )

    if uses_cutting_plane:
        return optimizer.solve_iterative(
            moments=moments,
            current_weights=current_weights,
            current_cash=current_cash,
            inputs=inputs,
            max_iter=max_iter,
            **solver_options,
        )

    return optimizer.solve(
        moments=moments,
        current_weights=current_weights,
        current_cash=current_cash,
        inputs=inputs,
        **solver_options,
    )
