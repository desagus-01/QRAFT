import pytest

from qraft.backtest.selection.overlays import apply_problem_hyperparameters
from qraft.backtest.selection.results import PolicyParams
from qraft.construction.optimization.constraints import LongOnly
from qraft.construction.optimization.objectives.specs import (
    CashReturn,
    CovarianceRisk,
    ExpectedReturn,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from qraft.construction.optimization.problem import MPOProblem, MPOProblemBuilder


def test_updates_preset_problem_hyperparameters_without_mutating_original():
    constraint = LongOnly()
    problem = MPOProblem.preset(
        "mean_covariance",
        risk_aversion=1.0,
        alpha=0.05,
        constraints=(constraint,),
        allow_borrow=True,
        max_iter=123,
        transaction_cost_weight=0.5,
        solver="CLARABEL",
    )

    updated = apply_problem_hyperparameters(
        problem,
        PolicyParams.of(
            risk_aversion=2.0,
            transaction_cost_weight=0.75,
            holding_cost_weight=0.25,
        ),
    )

    assert problem.objective.terms[2].weight == 1.0
    assert problem.objective.terms[3].weight == 0.5
    assert problem.objective.terms[4].weight == 1.0
    assert updated.objective.terms[2].weight == 2.0
    assert updated.objective.terms[3].weight == 0.75
    assert updated.objective.terms[4].weight == 0.25
    assert updated.constraints == (constraint,)
    assert updated.allow_borrow is True
    assert updated.max_iter == 123
    assert updated.solver_options == {"solver": "CLARABEL"}


def test_updates_explicit_problem_objective():
    problem = (
        MPOProblemBuilder()
        .add(ExpectedReturn())
        .add(CashReturn())
        .add(CovarianceRisk(), weight=1.0)
        .add(TransactionCost(), weight=0.5)
        .build()
    )

    updated = apply_problem_hyperparameters(
        problem,
        PolicyParams.of(risk_aversion=3.0, transaction_cost_weight=0.2),
    )

    assert problem.objective is not None
    assert updated.objective is not None
    assert problem.objective.terms[2].weight == 1.0
    assert problem.objective.terms[3].weight == 0.5
    assert updated.objective.terms[2].weight == 3.0
    assert updated.objective.terms[3].weight == 0.2
    assert updated.constraints == problem.constraints
    assert updated.max_iter == problem.max_iter


def test_explicit_problem_preserves_non_objective_settings():
    constraint = LongOnly()
    problem = MPOProblem(
        objective=ObjectiveSpec(
            terms=(
                WeightedTerm(1.0, ExpectedReturn()),
                WeightedTerm(2.0, CovarianceRisk()),
            )
        ),
        constraints=(constraint,),
        allow_borrow=True,
        max_iter=321,
        solver_options={"solver": "CLARABEL"},
    )

    updated = apply_problem_hyperparameters(
        problem,
        PolicyParams.of(risk_aversion=4.0),
    )

    assert updated.objective is not None
    assert updated.objective.terms[1].weight == 4.0
    assert updated.constraints == (constraint,)
    assert updated.allow_borrow is True
    assert updated.max_iter == 321
    assert updated.solver_options == {"solver": "CLARABEL"}


def test_unsupported_preset_parameter_fails():
    problem = MPOProblem.preset("mean_covariance", risk_aversion=1.0)

    with pytest.raises(ValueError, match="Unsupported objective hyperparameter"):
        apply_problem_hyperparameters(problem, PolicyParams.of(max_weight=0.1))


def test_unsupported_explicit_parameter_fails():
    problem = MPOProblem(
        objective=ObjectiveSpec(terms=(WeightedTerm(1.0, ExpectedReturn()),))
    )

    with pytest.raises(ValueError, match="Unsupported objective hyperparameter"):
        apply_problem_hyperparameters(problem, PolicyParams.of(max_weight=0.1))
