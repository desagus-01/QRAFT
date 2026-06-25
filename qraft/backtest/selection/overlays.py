from __future__ import annotations

from dataclasses import replace
from typing import Any

from qraft.backtest.selection.results import PolicyParams
from qraft.construction.optimization.objectives.specs import (
    CovarianceRisk,
    CVaRCuttingPlane,
    CVaRRisk,
    HoldingCost,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from qraft.construction.optimization.problem import MPOProblem

_RISK_SPECS = (CovarianceRisk, CVaRRisk, CVaRCuttingPlane)
_ALPHA_SPECS = (CVaRRisk, CVaRCuttingPlane)
_OBJECTIVE_PARAMS = {
    "risk_aversion",
    "transaction_cost_weight",
    "holding_cost_weight",
    "alpha",
}


def apply_problem_hyperparameters(
    problem: MPOProblem,
    params: PolicyParams,
) -> MPOProblem:
    """Apply hyperparameter overrides to preset or explicit MPO problems."""
    if problem._preset is not None:
        return _apply_preset_hyperparameters(problem, params)
    if problem.objective is not None:
        return replace(
            problem,
            objective=apply_objective_hyperparameters(problem.objective, params),
        )
    raise ValueError("MPOProblem must define either a preset or an objective")


def apply_objective_hyperparameters(
    objective: ObjectiveSpec,
    params: PolicyParams,
) -> ObjectiveSpec:
    """Apply semantic hyperparameter overrides to an objective spec."""
    values = params.as_dict()
    unknown = set(values) - _OBJECTIVE_PARAMS
    if unknown:
        raise ValueError(f"Unsupported objective hyperparameter(s): {sorted(unknown)}")

    terms = objective.terms
    if "risk_aversion" in values:
        terms = _replace_unique_weight(
            terms,
            _RISK_SPECS,
            values["risk_aversion"],
            name="risk_aversion",
        )
    if "transaction_cost_weight" in values:
        terms = _replace_unique_weight(
            terms,
            (TransactionCost,),
            values["transaction_cost_weight"],
            name="transaction_cost_weight",
        )
    if "holding_cost_weight" in values:
        terms = _replace_unique_weight(
            terms,
            (HoldingCost,),
            values["holding_cost_weight"],
            name="holding_cost_weight",
        )
    if "alpha" in values:
        terms = _replace_unique_spec_field(
            terms,
            _ALPHA_SPECS,
            field="alpha",
            value=values["alpha"],
            name="alpha",
        )

    return replace(objective, terms=terms)


def _apply_preset_hyperparameters(
    problem: MPOProblem,
    params: PolicyParams,
) -> MPOProblem:
    values = params.as_dict()
    unknown = set(values) - _OBJECTIVE_PARAMS
    if unknown:
        raise ValueError(f"Unsupported preset hyperparameter(s): {sorted(unknown)}")

    assert problem._preset is not None
    return replace(problem, _preset=replace(problem._preset, **values))


def _replace_unique_weight(
    terms: tuple[WeightedTerm, ...],
    spec_types: tuple[type[Any], ...],
    weight: Any,
    *,
    name: str,
) -> tuple[WeightedTerm, ...]:
    index = _unique_term_index(terms, spec_types, name=name)
    return tuple(
        replace(term, weight=weight) if i == index else term
        for i, term in enumerate(terms)
    )


def _replace_unique_spec_field(
    terms: tuple[WeightedTerm, ...],
    spec_types: tuple[type[Any], ...],
    *,
    field: str,
    value: Any,
    name: str,
) -> tuple[WeightedTerm, ...]:
    index = _unique_term_index(terms, spec_types, name=name)
    return tuple(
        replace(term, spec=replace(term.spec, **{field: value})) if i == index else term
        for i, term in enumerate(terms)
    )


def _unique_term_index(
    terms: tuple[WeightedTerm, ...],
    spec_types: tuple[type[Any], ...],
    *,
    name: str,
) -> int:
    indices = [i for i, term in enumerate(terms) if isinstance(term.spec, spec_types)]
    if not indices:
        type_names = ", ".join(t.__name__ for t in spec_types)
        raise ValueError(f"Cannot tune {name}: no {type_names} term found")
    if len(indices) > 1:
        type_names = ", ".join(t.__name__ for t in spec_types)
        raise ValueError(f"Cannot tune {name}: multiple {type_names} terms found")
    return indices[0]
