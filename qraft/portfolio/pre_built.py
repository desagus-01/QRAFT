from typing import Literal

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


def mean_covariance_objectives(
    risk_aversion: float,
) -> ObjectiveSpec:
    return ObjectiveSpec(
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


def cvar_classical_objectives(
    cvar_aversion: float,
    alpha: float = 0.05,
) -> ObjectiveSpec:
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(cvar_aversion, CVaRRisk(alpha=alpha)),
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


def cvar_cuts_objectives(
    cvar_aversion: float,
    alpha: float = 0.05,
) -> ObjectiveSpec:
    return ObjectiveSpec(
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


PreMadeObjectives = Literal["mean_covariance", "cvar_classic", "cvar_cuts"]


def _map_type_to_objective(
    type: PreMadeObjectives,
    risk_aversion: float,
    cvar_alpha: float | None,
) -> ObjectiveSpec:
    if type == "mean_covariance":
        objective = mean_covariance_objectives(risk_aversion=risk_aversion)
    elif type == "cvar_classic" and cvar_alpha is not None:
        objective = cvar_classical_objectives(
            cvar_aversion=risk_aversion, alpha=cvar_alpha
        )
    elif type == "cvar_cuts" and cvar_alpha is not None:
        objective = cvar_cuts_objectives(cvar_aversion=risk_aversion, alpha=cvar_alpha)
    else:
        raise ValueError(f"Your {type} is not valid, please choose a suitable one.")
    return objective


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
):
    objective = _map_type_to_objective(
        type=type, risk_aversion=risk_aversion, cvar_alpha=cvar_alpha
    )

    if type == "cvar_cuts":
        return MultiPeriodOptimizer(
            objective=objective,
            horizons=horizons,
            n_assets=n_assets,
            constraints=constraints,
            n_scenarios=moments.scenario_returns.shape[0],
        ).solve_iterative(moments, current_weights, max_iter=max_iter, **solver_options)
    else:
        return MultiPeriodOptimizer(
            objective=objective,
            horizons=horizons,
            n_assets=n_assets,
            constraints=constraints,
            n_scenarios=moments.scenario_returns.shape[0],
        ).solve(
            moments=moments,
            current_weights=current_weights,
            **solver_options,
        )
