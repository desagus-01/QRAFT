import pytest

from qraft.construction.optimization.objectives.specs import (
    CVaRCuttingPlane,
    CVaRRisk,
    CashReturn,
    CovarianceRisk,
    ExpectedReturn,
    HoldingCost,
    TransactionCost,
)
from qraft.construction.optimization.presets import (
    build_preset_objective,
    cvar_scale_requires_cuts,
    resolve_cvar_auto,
)


def _spec_types(objective):
    return tuple(type(term.spec) for term in objective.terms)


def test_mean_covariance_preset_includes_default_terms_and_costs() -> None:
    objective, cvar_auto = build_preset_objective("mean_covariance", 2.0)

    assert cvar_auto is False
    assert _spec_types(objective) == (
        ExpectedReturn,
        CashReturn,
        CovarianceRisk,
        TransactionCost,
        HoldingCost,
    )
    assert objective.terms[2].weight == 2.0


def test_preset_costs_none_omits_transaction_and_holding_costs() -> None:
    objective, _ = build_preset_objective("mean_covariance", 2.0, costs="none")

    assert _spec_types(objective) == (ExpectedReturn, CashReturn, CovarianceRisk)


def test_preset_rejects_unknown_cost_mode() -> None:
    with pytest.raises(ValueError, match="costs must be"):
        build_preset_objective("mean_covariance", 2.0, costs="cheap")  # type: ignore[arg-type]


def test_cvar_presets_select_expected_formulation() -> None:
    classic, classic_auto = build_preset_objective("cvar_classic", 4.0, alpha=0.1)
    cuts, cuts_auto = build_preset_objective("cvar_cuts", 4.0, alpha=0.1)
    auto, auto_flag = build_preset_objective("cvar_auto", 4.0, alpha=0.1)

    assert classic_auto is False
    assert cuts_auto is False
    assert auto_flag is True
    assert CVaRRisk in _spec_types(classic)
    assert CVaRCuttingPlane in _spec_types(cuts)
    assert CVaRCuttingPlane in _spec_types(auto)


def test_cvar_presets_require_alpha() -> None:
    with pytest.raises(ValueError, match="requires a CVaR alpha"):
        build_preset_objective("cvar_classic", 4.0, alpha=None)


def test_resolve_cvar_auto_replaces_cuts_for_small_problem() -> None:
    objective, _ = build_preset_objective("cvar_auto", 4.0, alpha=0.07, costs="none")

    resolved = resolve_cvar_auto(objective, horizons=2, n_scenarios=10)

    cvar_terms = [term for term in resolved.terms if isinstance(term.spec, CVaRRisk)]
    assert len(cvar_terms) == 1
    assert cvar_terms[0].spec.alpha == 0.07
    assert ExpectedReturn in _spec_types(resolved)


def test_resolve_cvar_auto_keeps_cuts_for_large_problem() -> None:
    objective, _ = build_preset_objective("cvar_auto", 4.0, alpha=0.07, costs="none")

    assert resolve_cvar_auto(objective, horizons=10, n_scenarios=100) is objective


def test_cvar_scale_requires_cuts_boundary() -> None:
    assert cvar_scale_requires_cuts(horizons=10, n_scenarios=99) is False
    assert cvar_scale_requires_cuts(horizons=10, n_scenarios=100) is True
