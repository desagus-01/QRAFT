from __future__ import annotations

import cvxpy as cp

from qraft.construction.optimization.optimization import MultiPeriodOptimizer
from qraft.construction.optimization.objectives.specs import (
    ExpectedReturn,
    ObjectiveSpec,
    WeightedTerm,
)


def test_optimal_inaccurate_is_failure_not_success() -> None:
    optimizer = MultiPeriodOptimizer(
        objective=ObjectiveSpec((WeightedTerm(1.0, ExpectedReturn()),)),
        horizons=1,
        n_assets=1,
        n_scenarios=1,
    )
    optimizer.problem = cp.Problem(cp.Maximize(0))
    optimizer.problem._status = "optimal_inaccurate"

    failure = optimizer._failure_if_not_optimal(raise_on_failure=False)

    assert failure is not None
    assert failure.status == "optimal_inaccurate"
