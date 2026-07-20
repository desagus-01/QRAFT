from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.optimization.optimization import MultiPeriodOptimizer
from qraft.construction.optimization.objectives.specs import (
    CVaRCuttingPlane,
    CVaRRisk,
    ExpectedReturn,
    ObjectiveSpec,
    WeightedTerm,
)


def test_optimal_inaccurate_is_success_not_failure() -> None:
    optimizer = MultiPeriodOptimizer(
        objective=ObjectiveSpec((WeightedTerm(1.0, ExpectedReturn()),)),
        horizons=1,
        n_assets=1,
        n_scenarios=1,
    )
    optimizer.problem = cp.Problem(cp.Maximize(0))
    optimizer.problem._status = "optimal_inaccurate"

    with pytest.warns(RuntimeWarning, match="optimal_inaccurate") as warnings:
        failure = optimizer._failure_if_not_optimal(raise_on_failure=False)

    assert failure is None
    assert len(warnings) == 1


def test_unbounded_cvar_cutting_plane_falls_back_to_ru_lp(caplog) -> None:
    inputs = OptimizerInputs.from_arrays(
        assets=["A"],
        mean=np.array([[0.01]]),
        scenario_returns=np.array([[[-0.2]], [[0.1]], [[0.2]]]),
        scenario_probs=np.full(3, 1.0 / 3.0),
    )
    current_weights = np.array([0.0])
    cuts_objective = ObjectiveSpec(
        (
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(1.0, CVaRCuttingPlane(alpha=0.5)),
        )
    )
    classic_objective = ObjectiveSpec(
        (
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(1.0, CVaRRisk(alpha=0.5)),
        )
    )
    cuts = MultiPeriodOptimizer(
        objective=cuts_objective,
        horizons=1,
        n_assets=1,
        n_scenarios=3,
        allow_borrow=True,
    )
    classic = MultiPeriodOptimizer(
        objective=classic_objective,
        horizons=1,
        n_assets=1,
        n_scenarios=3,
        allow_borrow=True,
    )

    fallback_result = cuts.solve_auto(inputs, current_weights, current_cash=1.0)
    classic_result = classic.solve(inputs, current_weights, current_cash=1.0)

    assert fallback_result.is_optimal
    assert classic_result.is_optimal
    np.testing.assert_allclose(
        fallback_result.planned_weights,
        classic_result.planned_weights,
        atol=1e-7,
    )
    assert fallback_result.objective_value == pytest.approx(
        classic_result.objective_value, abs=1e-8
    )
    assert any(
        getattr(record, "qraft_event", None) == "optimization.cvar_fallback"
        for record in caplog.records
    )
