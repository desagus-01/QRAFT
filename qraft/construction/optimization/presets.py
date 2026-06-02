from typing import Literal

from construction.optimization.objectives.specs import (
    CashReturn,
    CovarianceRisk,
    CVaRCuttingPlane,
    CVaRRisk,
    ExpectedReturn,
    HoldingCost,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)

PreMadeObjectives = Literal["mean_covariance", "cvar_auto", "cvar_classic", "cvar_cuts"]


def _default_transaction_cost() -> TransactionCost:
    return TransactionCost(
        cost=0.0005,
        pershare_cost=0.005,
        market_impact=0.3,
        exponent=1.5,
        c_bias=0.0003,
    )


def _default_holding_cost() -> HoldingCost:
    return HoldingCost(
        short_fees=0.0,
        long_fees=0.00119,
        dividends=0.0,
        periods_per_year=252,
    )


def _mean_covariance_objectives(
    risk_aversion: float,
    *,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
) -> ObjectiveSpec:
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn(decay=0.9)),
            WeightedTerm(1.0, CashReturn()),
            WeightedTerm(risk_aversion, CovarianceRisk()),
            WeightedTerm(
                transaction_cost_weight,
                transaction_cost or _default_transaction_cost(),
            ),
            WeightedTerm(holding_cost_weight, holding_cost or _default_holding_cost()),
        )
    )


def _cvar_classical_objectives(
    cvar_aversion: float,
    alpha: float = 0.05,
    *,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
) -> ObjectiveSpec:
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(1.0, CashReturn()),
            WeightedTerm(cvar_aversion, CVaRRisk(alpha=alpha)),
            WeightedTerm(
                transaction_cost_weight,
                transaction_cost or _default_transaction_cost(),
            ),
            WeightedTerm(holding_cost_weight, holding_cost or _default_holding_cost()),
        )
    )


def _cvar_cuts_objectives(
    cvar_aversion: float,
    alpha: float = 0.05,
    *,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
) -> ObjectiveSpec:
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(1.0, CashReturn()),
            WeightedTerm(cvar_aversion, CVaRCuttingPlane(alpha=alpha)),
            WeightedTerm(
                transaction_cost_weight,
                transaction_cost or _default_transaction_cost(),
            ),
            WeightedTerm(holding_cost_weight, holding_cost or _default_holding_cost()),
        )
    )


def _select_cvar_solver(
    horizons: int, n_scenarios: int, problem_limit: int = 1_000
) -> PreMadeObjectives:
    problem_scale = horizons * n_scenarios
    return "cvar_cuts" if problem_scale >= problem_limit else "cvar_classic"


def _build_preset_objective(
    objective_type: PreMadeObjectives,
    risk_aversion: float,
    *,
    alpha: float | None = 0.05,
    horizons: int,
    n_scenarios: int,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
) -> ObjectiveSpec:
    resolved_objective_type: PreMadeObjectives = (
        _select_cvar_solver(horizons=horizons, n_scenarios=n_scenarios)
        if objective_type == "cvar_auto"
        else objective_type
    )

    if resolved_objective_type == "mean_covariance":
        return _mean_covariance_objectives(
            risk_aversion=risk_aversion,
            transaction_cost=transaction_cost,
            transaction_cost_weight=transaction_cost_weight,
            holding_cost=holding_cost,
            holding_cost_weight=holding_cost_weight,
        )
    if resolved_objective_type == "cvar_classic" and alpha is not None:
        return _cvar_classical_objectives(
            cvar_aversion=risk_aversion,
            alpha=alpha,
            transaction_cost=transaction_cost,
            transaction_cost_weight=transaction_cost_weight,
            holding_cost=holding_cost,
            holding_cost_weight=holding_cost_weight,
        )
    if resolved_objective_type == "cvar_cuts" and alpha is not None:
        return _cvar_cuts_objectives(
            cvar_aversion=risk_aversion,
            alpha=alpha,
            transaction_cost=transaction_cost,
            transaction_cost_weight=transaction_cost_weight,
            holding_cost=holding_cost,
            holding_cost_weight=holding_cost_weight,
        )
    raise ValueError(
        f"Your {objective_type} is not valid, please choose a suitable one."
    )
