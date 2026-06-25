import pytest

from qraft.backtest.selection.overlays import apply_objective_hyperparameters
from qraft.backtest.selection.results import PolicyParams
from qraft.construction.optimization.objectives.specs import (
    CVaRCuttingPlane,
    CVaRRisk,
    CashReturn,
    CovarianceRisk,
    ExpectedReturn,
    HoldingCost,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)


def _mean_variance_objective() -> ObjectiveSpec:
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(1.0, CashReturn()),
            WeightedTerm(2.0, CovarianceRisk()),
            WeightedTerm(0.5, TransactionCost()),
            WeightedTerm(0.25, HoldingCost(0.0, 0.001, 0.0)),
        )
    )


def test_updates_covariance_risk_weight_without_mutating_original():
    objective = _mean_variance_objective()

    updated = apply_objective_hyperparameters(
        objective,
        PolicyParams.of(risk_aversion=3.0),
    )

    assert objective.terms[2].weight == 2.0
    assert updated.terms[2].weight == 3.0
    assert isinstance(updated.terms[2].spec, CovarianceRisk)


def test_updates_transaction_and_holding_cost_weights():
    objective = _mean_variance_objective()

    updated = apply_objective_hyperparameters(
        objective,
        PolicyParams.of(transaction_cost_weight=0.75, holding_cost_weight=0.1),
    )

    assert updated.terms[3].weight == 0.75
    assert updated.terms[4].weight == 0.1


def test_updates_cvar_weight_and_alpha():
    objective = ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(4.0, CVaRRisk(alpha=0.05)),
        )
    )

    updated = apply_objective_hyperparameters(
        objective,
        PolicyParams.of(risk_aversion=5.0, alpha=0.01),
    )

    assert objective.terms[1].weight == 4.0
    assert objective.terms[1].spec.alpha == 0.05
    assert updated.terms[1].weight == 5.0
    assert updated.terms[1].spec.alpha == 0.01


def test_updates_cutting_plane_alpha():
    objective = ObjectiveSpec(terms=(WeightedTerm(4.0, CVaRCuttingPlane(alpha=0.05)),))

    updated = apply_objective_hyperparameters(objective, PolicyParams.of(alpha=0.02))

    assert updated.terms[0].spec.alpha == 0.02


def test_unknown_parameter_fails():
    with pytest.raises(ValueError, match="Unsupported objective hyperparameter"):
        apply_objective_hyperparameters(
            _mean_variance_objective(),
            PolicyParams.of(max_weight=0.1),
        )


def test_missing_matching_term_fails():
    objective = ObjectiveSpec(terms=(WeightedTerm(1.0, ExpectedReturn()),))

    with pytest.raises(ValueError, match="no CovarianceRisk"):
        apply_objective_hyperparameters(
            objective,
            PolicyParams.of(risk_aversion=2.0),
        )


def test_duplicate_matching_terms_fail():
    objective = ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, CovarianceRisk()),
            WeightedTerm(2.0, CVaRRisk(alpha=0.05)),
        )
    )

    with pytest.raises(ValueError, match="multiple CovarianceRisk"):
        apply_objective_hyperparameters(
            objective,
            PolicyParams.of(risk_aversion=3.0),
        )
