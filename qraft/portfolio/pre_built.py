import numpy as np
from numpy.typing import NDArray
from portfolio.policy.constraints import PortfolioConstraint
from portfolio.policy.moments import HorizonMoments
from portfolio.policy.objectives.specs import (
    CovarianceRisk,
    CVaRCuttingPlane,
    CVaRRisk,
    ExpectedReturn,
    HoldingCost,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from portfolio.policy.optimization import MultiPeriodOptimizer


def classic_mpo(
    horizons: int,
    n_assets: int,
    risk_aversion: float,
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    transaction_cost: float = 0.01,
    constraints: list[PortfolioConstraint] | None = None,
    **solver_options,
):
    objective = ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn(decay=0.9)),
            WeightedTerm(risk_aversion, CovarianceRisk()),
            WeightedTerm(
                1.0,
                TransactionCost(
                    cost=0.0005,
                    pershare_cost=0.005,
                    market_impact=0.8,
                    exponent=1.5,
                    c_bias=0.0003,
                ),
            ),
            WeightedTerm(
                1.0,
                HoldingCost(
                    short_fees=0.0,
                    long_fees=0.3,
                    dividends=0.0,
                    periods_per_year=252,
                ),
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
    moments: HorizonMoments,
    current_weights: NDArray[np.floating],
    transaction_cost: float = 0.01,
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
                1.0,
                TransactionCost(
                    cost=0.0005,
                    pershare_cost=0.005,
                    market_impact=0.8,
                    exponent=1.5,
                    c_bias=0.0003,
                ),
            ),
            WeightedTerm(
                1.0,
                HoldingCost(
                    short_fees=0.0,
                    long_fees=0.3,
                    dividends=0.0,
                    periods_per_year=252,
                ),
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
