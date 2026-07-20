from dataclasses import replace
from typing import Literal

from qraft.construction.optimization.objectives.specs import (
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

_CVAR_CUTS_SCALE_LIMIT = 1_000
PreMadeObjectives = Literal["mean_covariance", "cvar_auto", "cvar_classic", "cvar_cuts"]
PresetCosts = Literal["default", "none"]


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
    risk_aversion: float | None,
    *,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
    costs: PresetCosts = "default",
) -> ObjectiveSpec:
    cost_terms = ()
    if costs == "default":
        cost_terms = (
            WeightedTerm(
                transaction_cost_weight,
                transaction_cost or _default_transaction_cost(),
            ),
            WeightedTerm(holding_cost_weight, holding_cost or _default_holding_cost()),
        )
    elif costs != "none":
        raise ValueError("costs must be 'default' or 'none'")
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn(decay=0.9)),
            WeightedTerm(1.0, CashReturn()),
            WeightedTerm(risk_aversion, CovarianceRisk()),
            *cost_terms,
        )
    )


def _cvar_classical_objectives(
    cvar_aversion: float | None,
    alpha: float = 0.05,
    *,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
    costs: PresetCosts = "default",
) -> ObjectiveSpec:
    cost_terms = ()
    if costs == "default":
        cost_terms = (
            WeightedTerm(
                transaction_cost_weight,
                transaction_cost or _default_transaction_cost(),
            ),
            WeightedTerm(holding_cost_weight, holding_cost or _default_holding_cost()),
        )
    elif costs != "none":
        raise ValueError("costs must be 'default' or 'none'")
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(1.0, CashReturn()),
            WeightedTerm(cvar_aversion, CVaRRisk(alpha=alpha)),
            *cost_terms,
        )
    )


def _cvar_cuts_objectives(
    cvar_aversion: float | None,
    alpha: float = 0.05,
    *,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
    costs: PresetCosts = "default",
) -> ObjectiveSpec:
    cost_terms = ()
    if costs == "default":
        cost_terms = (
            WeightedTerm(
                transaction_cost_weight,
                transaction_cost or _default_transaction_cost(),
            ),
            WeightedTerm(holding_cost_weight, holding_cost or _default_holding_cost()),
        )
    elif costs != "none":
        raise ValueError("costs must be 'default' or 'none'")
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(1.0, CashReturn()),
            WeightedTerm(cvar_aversion, CVaRCuttingPlane(alpha=alpha)),
            *cost_terms,
        )
    )


def cvar_scale_requires_cuts(
    horizons: int, n_scenarios: int, limit: int = _CVAR_CUTS_SCALE_LIMIT
) -> bool:
    """True when the problem is large enough to prefer the cutting-plane CVaR."""
    return horizons * n_scenarios >= limit


def build_preset_objective(
    objective_type: PreMadeObjectives,
    risk_aversion: float | None,
    *,
    alpha: float | None = 0.05,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
    costs: PresetCosts = "default",
) -> tuple[ObjectiveSpec, bool]:
    """Build a preset objective eagerly.

    Returns ``(objective, cvar_auto)``. ``cvar_auto`` flags that the CVaR
    formulation should be re-selected from the problem size at compile time
    (see :func:`resolve_cvar_auto`); for every other preset it is ``False`` and
    the returned objective is final.
    """
    if objective_type == "mean_covariance":
        return (
            _mean_covariance_objectives(
                risk_aversion=risk_aversion,
                transaction_cost=transaction_cost,
                transaction_cost_weight=transaction_cost_weight,
                holding_cost=holding_cost,
                holding_cost_weight=holding_cost_weight,
                costs=costs,
            ),
            False,
        )
    if objective_type not in ("cvar_classic", "cvar_cuts", "cvar_auto"):
        raise ValueError(
            f"Your {objective_type} is not valid, please choose a suitable one."
        )
    if alpha is None:
        raise ValueError(f"{objective_type} requires a CVaR alpha; got alpha=None.")

    use_cuts = objective_type in ("cvar_cuts", "cvar_auto")
    builder = _cvar_cuts_objectives if use_cuts else _cvar_classical_objectives
    objective = builder(
        cvar_aversion=risk_aversion,
        alpha=alpha,
        transaction_cost=transaction_cost,
        transaction_cost_weight=transaction_cost_weight,
        holding_cost=holding_cost,
        holding_cost_weight=holding_cost_weight,
        costs=costs,
    )
    return objective, objective_type == "cvar_auto"


def resolve_cvar_auto(
    objective: ObjectiveSpec, horizons: int, n_scenarios: int
) -> ObjectiveSpec:
    """Pick the CVaR formulation for a ``cvar_auto`` objective from problem size.

    Large problems keep the cutting-plane term; small ones are downgraded to the
    classic Rockafellar-Uryasev LP. Non-CVaR terms are left untouched.
    """
    if cvar_scale_requires_cuts(horizons, n_scenarios):
        return objective
    terms = tuple(
        replace(term, spec=CVaRRisk(alpha=term.spec.alpha))
        if isinstance(term.spec, CVaRCuttingPlane)
        else term
        for term in objective.terms
    )
    return replace(objective, terms=terms)
