from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from qraft.construction.optimization.constraints import PortfolioConstraint
from qraft.construction.optimization.objectives.specs import (
    HoldingCost,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from qraft.construction.optimization.optimization import (
    MultiPeriodOptimizer,
)
from qraft.construction.optimization.presets import (
    PreMadeObjectives,
    build_preset_objective,
    resolve_cvar_auto,
)


@dataclass(frozen=True, slots=True)
class MPOProblem:
    objective: ObjectiveSpec
    cvar_auto: bool = False

    constraints: tuple[PortfolioConstraint, ...] = ()
    allow_borrow: bool = False
    max_iter: int = 200
    solver_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def preset(
        cls,
        objective_type: PreMadeObjectives,
        risk_aversion: float,
        *,
        alpha: float | None = 0.05,
        constraints: Sequence[PortfolioConstraint] = (),
        allow_borrow: bool = False,
        max_iter: int = 200,
        transaction_cost: TransactionCost | None = None,
        transaction_cost_weight: float = 1.0,
        holding_cost: HoldingCost | None = None,
        holding_cost_weight: float = 1.0,
        **solver_options: Any,
    ) -> "MPOProblem":
        objective, cvar_auto = build_preset_objective(
            objective_type,
            risk_aversion,
            alpha=alpha,
            transaction_cost=transaction_cost,
            transaction_cost_weight=transaction_cost_weight,
            holding_cost=holding_cost,
            holding_cost_weight=holding_cost_weight,
        )

        return cls(
            objective=objective,
            cvar_auto=cvar_auto,
            constraints=tuple(constraints),
            allow_borrow=allow_borrow,
            max_iter=max_iter,
            solver_options=solver_options,
        )

    def cost_specs(self) -> tuple[TransactionCost | None, HoldingCost | None]:
        """Return (transaction_cost, holding_cost) specs this problem penalises.

        For preset problems the defaults from ``_build_preset_objective`` are
        resolved so that ``CostModel.from_policy`` returns the specs the
        optimiser actually uses.
        """
        transaction: TransactionCost | None = None
        holding: HoldingCost | None = None
        for term in self.objective.terms:
            if isinstance(term.spec, TransactionCost):
                transaction = term.spec
            elif isinstance(term.spec, HoldingCost):
                holding = term.spec
        return transaction, holding

    def compile(
        self,
        horizons: int,
        n_assets: int,
        n_scenarios: int,
    ) -> MultiPeriodOptimizer:
        """
        Allocate CVXPY variables and return a compiled
        :class:`MultiPeriodOptimizer` ready to be solved.
        """
        objective = (
            resolve_cvar_auto(self.objective, horizons, n_scenarios)
            if self.cvar_auto
            else self.objective
        )
        return MultiPeriodOptimizer(
            objective=objective,
            horizons=horizons,
            n_assets=n_assets,
            n_scenarios=n_scenarios,
            constraints=self.constraints,
            allow_borrow=self.allow_borrow,
        )


@dataclass(frozen=True, slots=True)
class MPOProblemBuilder:
    """
    Immutable fluent builder for :class:`MPOProblem`.

    Each method returns a new builder with the updated setting applied,
    leaving the original unchanged.

    Examples
    --------
    >>> problem = (
    ...     MPOProblemBuilder()
    ...     .add(ExpectedReturn(decay=0.9))
    ...     .add(CovarianceRisk(), weight=2.0)
    ...     .add(TransactionCost())
    ...     .constrain(LongOnly(), TurnoverLimit(0.10))
    ...     .build()
    ... )
    """

    terms: tuple[WeightedTerm, ...] = ()
    constraints_: tuple[PortfolioConstraint, ...] = ()
    allow_borrow_: bool = False
    max_iter_: int = 200
    solver_options_: Mapping[str, Any] = field(default_factory=dict)

    def add(self, spec: Any, weight: float = 1.0) -> "MPOProblemBuilder":
        """Append a weighted objective term."""
        return MPOProblemBuilder(
            terms=(*self.terms, WeightedTerm(weight, spec)),
            constraints_=self.constraints_,
            allow_borrow_=self.allow_borrow_,
            max_iter_=self.max_iter_,
            solver_options_=self.solver_options_,
        )

    def constrain(self, *constraints: PortfolioConstraint) -> "MPOProblemBuilder":
        """Append one or more portfolio constraints."""
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=(*self.constraints_, *constraints),
            allow_borrow_=self.allow_borrow_,
            max_iter_=self.max_iter_,
            solver_options_=self.solver_options_,
        )

    def allow_borrow(self, value: bool = True) -> "MPOProblemBuilder":
        """Allow (or disallow) negative cash weights."""
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=self.constraints_,
            allow_borrow_=value,
            max_iter_=self.max_iter_,
            solver_options_=self.solver_options_,
        )

    def max_iter(self, value: int) -> "MPOProblemBuilder":
        """Set the maximum cutting-plane iterations."""
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=self.constraints_,
            allow_borrow_=self.allow_borrow_,
            max_iter_=value,
            solver_options_=self.solver_options_,
        )

    def solver_options(self, **options: Any) -> "MPOProblemBuilder":
        """Merge extra CVXPY solver options."""
        return MPOProblemBuilder(
            terms=self.terms,
            constraints_=self.constraints_,
            allow_borrow_=self.allow_borrow_,
            max_iter_=self.max_iter_,
            solver_options_={**self.solver_options_, **options},
        )

    def build(self) -> MPOProblem:
        """Finalise and return the :class:`MPOProblem`."""
        return MPOProblem(
            objective=ObjectiveSpec(terms=self.terms),
            constraints=self.constraints_,
            allow_borrow=self.allow_borrow_,
            max_iter=self.max_iter_,
            solver_options=self.solver_options_,
        )
