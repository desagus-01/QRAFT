import numpy as np
from numpy.typing import NDArray
from portfolio.policy.constraints import PortfolioConstraint
from portfolio.policy.moments import HorizonMoments
from portfolio.policy.objectives.specs import (
    CovarianceRisk,
    CVaRCuttingPlane,
    CVaRRisk,
    ExpectedReturn,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from portfolio.policy.optimization import MultiPeriodOptimizer


def classic_mpo(
    horizons: int,
    n_assets: int,
    risk_aversion: float,
    transaction_cost: float,
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    constraints: list[PortfolioConstraint] | None = None,
    **solver_options,
):
    objective = ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(risk_aversion, CovarianceRisk()),
            WeightedTerm(
                transaction_cost,
                TransactionCost(cost=1.0, market_impact=0.0, exponent=1.0),
            ),
        )
    )
    return MultiPeriodOptimizer(
        objective=objective,
        horizons=horizons,
        n_assets=n_assets,
        constraints=constraints,
        n_scenarios=moments.scenario_returns.shape[0],
    ).solve(moments, current_weights, inputs=None, **solver_options)


def cvar_mpo(
    horizons: int,
    n_assets: int,
    cvar_aversion: float,
    transaction_cost: float,
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    alpha: float = 0.05,
    constraints: list[PortfolioConstraint] | None = None,
    **solver_options,
):
    objective = ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(cvar_aversion, CVaRRisk(alpha=alpha)),
            WeightedTerm(
                transaction_cost,
                TransactionCost(cost=1.0, market_impact=0.0, exponent=1.0),
            ),
        )
    )
    return MultiPeriodOptimizer(
        objective=objective,
        horizons=horizons,
        n_assets=n_assets,
        constraints=constraints,
        n_scenarios=moments.scenario_returns.shape[0],
    ).solve(moments, current_weights, inputs=None, **solver_options)


def cvar_mpo_cuts(
    horizons: int,
    n_assets: int,
    cvar_aversion: float,
    transaction_cost: float,
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    alpha: float = 0.05,
    constraints: list[PortfolioConstraint] | None = None,
    max_iter: int = 200,
    **solver_options,
):
    objective = ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(cvar_aversion, CVaRCuttingPlane(alpha=alpha)),
            WeightedTerm(
                transaction_cost,
                TransactionCost(cost=1.0, market_impact=0.0, exponent=1.0),
            ),
        )
    )
    return MultiPeriodOptimizer(
        objective=objective,
        horizons=horizons,
        n_assets=n_assets,
        constraints=constraints,
        n_scenarios=moments.scenario_returns.shape[0],
    ).solve_iterative(moments, current_weights, max_iter=max_iter, **solver_options)
