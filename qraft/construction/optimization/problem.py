from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from construction.optimization.constraints import PortfolioConstraint
from construction.optimization.moments import HorizonMoments
from construction.optimization.objectives.specs import (
    CVaRCuttingPlane,
    HoldingCost,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from construction.optimization.optimization import MPOResult, MultiPeriodOptimizer
from construction.optimization.presets import PreMadeObjectives, build_preset_objective
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MPOProblem:
    """
    Size-agnostic multi-period optimization definition.

    The problem carries the objective and optimization settings without allocating
    CVXPY variables. ``compile`` binds it to a concrete horizon and asset universe.
    """

    objective: ObjectiveSpec
    constraints: tuple[PortfolioConstraint, ...] = ()
    allow_borrow: bool = False
    cvar_alpha: float | None = 0.05
    max_iter: int = 200
    solver_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def uses_cutting_plane(self) -> bool:
        return any(
            isinstance(weighted_term.spec, CVaRCuttingPlane)
            for weighted_term in self.objective.terms
        )

    def compile(
        self,
        horizons: int,
        n_assets: int,
        n_scenarios: int,
    ) -> MultiPeriodOptimizer:
        return MultiPeriodOptimizer(
            objective=self.objective,
            horizons=horizons,
            n_assets=n_assets,
            n_scenarios=n_scenarios,
            constraints=self.constraints,
            allow_borrow=self.allow_borrow,
        )

    def solve(
        self,
        *,
        horizons: int,
        n_assets: int,
        n_scenarios: int,
        moments: HorizonMoments,
        current_weights: NDArray[np.floating],
        current_cash: float | None = None,
        inputs: dict[str, Any] | None = None,
        **solver_options: Any,
    ) -> MPOResult:
        optimizer = self.compile(
            horizons=horizons,
            n_assets=n_assets,
            n_scenarios=n_scenarios,
        )
        options = {**self.solver_options, **solver_options}

        if self.uses_cutting_plane:
            return optimizer.solve_iterative(
                moments=moments,
                current_weights=current_weights,
                current_cash=current_cash,
                inputs=inputs,
                max_iter=self.max_iter,
                **options,
            )

        return optimizer.solve(
            moments=moments,
            current_weights=current_weights,
            current_cash=current_cash,
            inputs=inputs,
            **options,
        )


@dataclass(frozen=True, slots=True)
class MPOProblemBuilder:
    terms: tuple[WeightedTerm, ...] = ()
    constraints_: tuple[PortfolioConstraint, ...] = ()
    allow_borrow_: bool = False
    cvar_alpha_: float | None = 0.05
    max_iter_: int = 200
    solver_options_: Mapping[str, Any] = field(default_factory=dict)

    def add(self, spec: Any, weight: float = 1.0) -> "MPOProblemBuilder":
        return MPOProblemBuilder(
            terms=(*self.terms, WeightedTerm(weight, spec)),
            constraints_=self.constraints_,
            allow_borrow_=self.allow_borrow_,
            cvar_alpha_=self.cvar_alpha_,
            max_iter_=self.max_iter_,
            solver_options_=self.solver_options_,
        )

    def constrain(
        self,
        *constraints: PortfolioConstraint,
    ) -> "MPOProblemBuilder":
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=(*self.constraints_, *constraints),
            allow_borrow_=self.allow_borrow_,
            cvar_alpha_=self.cvar_alpha_,
            max_iter_=self.max_iter_,
            solver_options_=self.solver_options_,
        )

    def allow_borrow(self, value: bool = True) -> "MPOProblemBuilder":
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=self.constraints_,
            allow_borrow_=value,
            cvar_alpha_=self.cvar_alpha_,
            max_iter_=self.max_iter_,
            solver_options_=self.solver_options_,
        )

    def cvar_alpha(self, value: float | None) -> "MPOProblemBuilder":
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=self.constraints_,
            allow_borrow_=self.allow_borrow_,
            cvar_alpha_=value,
            max_iter_=self.max_iter_,
            solver_options_=self.solver_options_,
        )

    def max_iter(self, value: int) -> "MPOProblemBuilder":
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=self.constraints_,
            allow_borrow_=self.allow_borrow_,
            cvar_alpha_=self.cvar_alpha_,
            max_iter_=value,
            solver_options_=self.solver_options_,
        )

    def solver_options(self, **options: Any) -> "MPOProblemBuilder":
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=self.constraints_,
            allow_borrow_=self.allow_borrow_,
            cvar_alpha_=self.cvar_alpha_,
            max_iter_=self.max_iter_,
            solver_options_={**self.solver_options_, **options},
        )

    def build(self) -> MPOProblem:
        return MPOProblem(
            objective=ObjectiveSpec(terms=self.terms),
            constraints=self.constraints_,
            allow_borrow=self.allow_borrow_,
            cvar_alpha=self.cvar_alpha_,
            max_iter=self.max_iter_,
            solver_options=self.solver_options_,
        )


def build_preset_problem(
    objective_type: PreMadeObjectives,
    risk_aversion: float,
    *,
    alpha: float | None = 0.05,
    horizons: int,
    n_scenarios: int,
    constraints: Sequence[PortfolioConstraint] | None = None,
    allow_borrow: bool = False,
    max_iter: int = 200,
    transaction_cost: TransactionCost | None = None,
    transaction_cost_weight: float = 1.0,
    holding_cost: HoldingCost | None = None,
    holding_cost_weight: float = 1.0,
    **solver_options: Any,
) -> MPOProblem:
    objective = build_preset_objective(
        objective_type=objective_type,
        risk_aversion=risk_aversion,
        alpha=alpha,
        horizons=horizons,
        n_scenarios=n_scenarios,
        transaction_cost=transaction_cost,
        transaction_cost_weight=transaction_cost_weight,
        holding_cost=holding_cost,
        holding_cost_weight=holding_cost_weight,
    )
    return MPOProblem(
        objective=objective,
        constraints=tuple(constraints or ()),
        allow_borrow=allow_borrow,
        cvar_alpha=alpha,
        max_iter=max_iter,
        solver_options=solver_options,
    )
