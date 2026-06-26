import pytest

from qraft.backtest.selection import (
    PolicyParams,
    apply_hyperparameters,
    expand_candidates,
    param_grid,
)
from qraft.construction.optimization.objectives.specs import (
    CashReturn,
    CovarianceRisk,
    ExpectedReturn,
    ObjectiveSpec,
    WeightedTerm,
)
from qraft.construction.optimization.problem import MPOProblem
from qraft.construction.policies import EqualWeightPolicy, MPOPolicy


def test_param_grid_returns_cartesian_product_in_grid_order():
    params = tuple(
        param_grid(
            {
                "risk_aversion": [1.0, 2.0],
                "transaction_cost_weight": [0.5, 1.0],
            }
        )
    )

    assert [p.as_dict() for p in params] == [
        {"risk_aversion": 1.0, "transaction_cost_weight": 0.5},
        {"risk_aversion": 1.0, "transaction_cost_weight": 1.0},
        {"risk_aversion": 2.0, "transaction_cost_weight": 0.5},
        {"risk_aversion": 2.0, "transaction_cost_weight": 1.0},
    ]


def test_param_grid_empty_grid_returns_one_empty_param_set():
    assert tuple(param_grid({})) == (PolicyParams.of(),)


def test_param_grid_empty_values_return_no_param_sets():
    assert tuple(param_grid({"risk_aversion": []})) == ()


def test_apply_hyperparameters_overlays_preset_mpo_policy():
    policy = MPOPolicy.preset("mean_covariance", risk_aversion=1.0, name="base")

    variant = apply_hyperparameters(policy, PolicyParams.of(risk_aversion=2.0))

    assert isinstance(variant, MPOPolicy)
    assert variant is not policy
    assert policy.problem.objective.terms[2].weight == 1.0
    assert variant.problem.objective.terms[2].weight == 2.0
    assert variant.name == "base[risk_aversion=2]"


def test_apply_hyperparameters_overlays_explicit_mpo_policy():
    policy = MPOPolicy(
        problem=MPOProblem(
            objective=ObjectiveSpec(
                terms=(
                    WeightedTerm(1.0, ExpectedReturn()),
                    WeightedTerm(1.0, CashReturn()),
                    WeightedTerm(2.0, CovarianceRisk()),
                )
            )
        ),
        name="custom",
    )

    variant = apply_hyperparameters(policy, {"risk_aversion": 3.0})

    assert isinstance(variant, MPOPolicy)
    assert policy.problem.objective is not None
    assert variant.problem.objective is not None
    assert policy.problem.objective.terms[2].weight == 2.0
    assert variant.problem.objective.terms[2].weight == 3.0
    assert variant.name == "custom[risk_aversion=3]"


def test_apply_hyperparameters_preserves_base_name_for_empty_params():
    policy = MPOPolicy.preset("mean_covariance", risk_aversion=1.0, name="base")

    variant = apply_hyperparameters(policy, PolicyParams.of())

    assert isinstance(variant, MPOPolicy)
    assert variant.name == "base"


def test_apply_hyperparameters_does_not_share_optimizer_cache():
    policy = MPOPolicy.preset("mean_covariance", risk_aversion=1.0, name="base")
    policy._optimizer_cache[(("AAA",), 1, 2)] = object()  # type: ignore[assignment]

    variant = apply_hyperparameters(policy, PolicyParams.of(risk_aversion=2.0))

    assert isinstance(variant, MPOPolicy)
    assert variant._optimizer_cache == {}
    assert variant._optimizer_cache is not policy._optimizer_cache


def test_expand_candidates_returns_policy_candidates_with_provenance():
    policy = MPOPolicy.preset("mean_covariance", risk_aversion=1.0, name="base")

    candidates = expand_candidates(policy, {"risk_aversion": [0.5, 1.0, 2.0]})

    assert len(candidates) == 3
    assert [c.params.as_dict() for c in candidates] == [
        {"risk_aversion": 0.5},
        {"risk_aversion": 1.0},
        {"risk_aversion": 2.0},
    ]
    assert [c.policy.problem.objective.terms[2].weight for c in candidates] == [
        0.5,
        1.0,
        2.0,
    ]
    assert policy.problem.objective.terms[2].weight == 1.0


def test_non_mpo_policy_with_empty_params_returns_original_policy():
    policy = EqualWeightPolicy(target_cash_weight=0.0)

    variant = apply_hyperparameters(policy, PolicyParams.of())

    assert variant is policy


def test_non_mpo_policy_with_non_empty_params_fails():
    policy = EqualWeightPolicy(target_cash_weight=0.0)

    with pytest.raises(TypeError, match="does not expose tunable hyperparameters"):
        apply_hyperparameters(policy, PolicyParams.of(risk_aversion=1.0))
