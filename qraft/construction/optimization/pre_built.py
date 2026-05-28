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
    horizons: int,
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

    Parameters
    ----------
    objective_type:
        Named objective recipe.  ``"cvar_auto"`` is resolved automatically
        from problem size.
    horizons:
        Number of look-ahead periods.
    n_assets:
        Number of investable risky assets (excluding cash).
    risk_aversion:
        Scalar risk-aversion / CVaR-aversion coefficient.
    moments:
        Pre-computed :class:`HorizonMoments` for this solve.  Must contain
        ``cash_return`` when a ``CashReturn`` term is present in the objective.
    current_weights:
        Current risky-asset weight vector, shape ``(n_assets,)``.
    current_cash:
        Current cash weight.  If ``None``, inferred as
        ``1 - sum(current_weights)``.
    cvar_alpha:
        Tail probability for CVaR objectives.  Unused for
        ``"mean_covariance"``.
    constraints:
        Optional hard / soft portfolio constraints.
    inputs:
        Extra inputs forwarded to handler ``update`` calls (e.g.
        ``{"prices": ..., "volume": ...}``).
    allow_borrow:
        If ``False`` (default), the optimizer enforces ``cash >= 0`` at
        every horizon.
    max_iter:
        Maximum cutting-plane iterations when using ``CVaRCuttingPlane``.
    **solver_options:
        Passed directly to ``cvxpy.Problem.solve``.
    """
    optimizer = MultiPeriodOptimizer.from_pre_built(
        objective_type=objective_type,
        horizons=horizons,
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
