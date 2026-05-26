import numpy as np
from numpy.typing import NDArray
from portfolio.policy.constraints import PortfolioConstraint
from portfolio.policy.moments import HorizonMoments
from portfolio.policy.objectives.specs import CVaRCuttingPlane
from portfolio.policy.optimization import (
    MPOResult,
    MultiPeriodOptimizer,
    PreMadeObjectives,
)


def multi_period_optimization(
    type: PreMadeObjectives,
    horizons: int,
    n_assets: int,
    risk_aversion: float,
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    cvar_alpha: float | None = 0.05,
    constraints: list[PortfolioConstraint] | None = None,
    max_iter: int = 200,
    **solver_options,
) -> MPOResult:
    """
    Convenience wrapper around :class:`~portfolio.policy.optimization.MultiPeriodOptimizer`.


    Parameters
    ----------
    type:
        Objective recipe.  Ignored when *optimizer* is provided.
    horizons:
        Look-ahead periods.  Ignored when *optimizer* is provided.
    n_assets:
        Number of assets.  Ignored when *optimizer* is provided.
    risk_aversion:
        Risk / CVaR-aversion scalar.  Ignored when *optimizer* is provided.
    moments:
        Forecast moments for the current time step.
    current_weights:
        Portfolio weights at the start of the current time step.
    cvar_alpha:
        CVaR tail probability.  Ignored when *optimizer* is provided.
    constraints:
        Portfolio constraints.  Ignored when *optimizer* is provided.
    max_iter:
        Max cutting-plane iterations (CVaR-cuts only).
    **solver_options:
        Forwarded verbatim to the underlying CVXPY solver.
    """
    optimizer = MultiPeriodOptimizer.from_pre_built(
        type=type,
        horizons=horizons,
        n_assets=n_assets,
        n_scenarios=moments.scenario_returns.shape[0],
        risk_aversion=risk_aversion,
        cvar_alpha=cvar_alpha,
        constraints=constraints,
    )

    uses_cutting_plane = any(
        isinstance(term.spec, CVaRCuttingPlane) for term in optimizer.objective.terms
    )

    if uses_cutting_plane:
        return optimizer.solve_iterative(
            moments, current_weights, max_iter=max_iter, **solver_options
        )
    return optimizer.solve(
        moments=moments, current_weights=current_weights, **solver_options
    )
