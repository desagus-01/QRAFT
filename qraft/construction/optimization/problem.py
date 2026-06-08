import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from construction.optimization.constraints import PortfolioConstraint
from construction.optimization.moments import HorizonMoments
from construction.optimization.objectives.specs import (
    HoldingCost,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)
from construction.optimization.optimization import MPOResult, MultiPeriodOptimizer
from construction.optimization.presets import PreMadeObjectives, _build_preset_objective
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PresetSpec:
    """
    Stores the parameters needed to build a preset ``ObjectiveSpec`` at
    compile time
    """

    objective_type: PreMadeObjectives
    risk_aversion: float
    alpha: float | None
    transaction_cost: TransactionCost | None
    transaction_cost_weight: float
    holding_cost: HoldingCost | None
    holding_cost_weight: float


@dataclass(frozen=True, slots=True)
class MPOProblem:
    objective: ObjectiveSpec | None = None
    _preset: _PresetSpec | None = field(default=None, repr=False)

    constraints: tuple[PortfolioConstraint, ...] = ()
    allow_borrow: bool = False
    max_iter: int = 200
    solver_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_objective = self.objective is not None
        has_preset = self._preset is not None
        if has_objective == has_preset:  # both set, or neither set
            raise ValueError(
                "Exactly one of `objective` or `_preset` must be provided. "
                "Use MPOProblem.preset() or MPOProblemBuilder.build()."
            )

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
        return cls(
            _preset=_PresetSpec(
                objective_type=objective_type,
                risk_aversion=risk_aversion,
                alpha=alpha,
                transaction_cost=transaction_cost,
                transaction_cost_weight=transaction_cost_weight,
                holding_cost=holding_cost,
                holding_cost_weight=holding_cost_weight,
            ),
            constraints=tuple(constraints),
            allow_borrow=allow_borrow,
            max_iter=max_iter,
            solver_options=solver_options,
        )

    def _resolve_objective(self, horizons: int, n_scenarios: int) -> ObjectiveSpec:
        """
        Return the concrete ``ObjectiveSpec`` for the given problem size.
        """
        if self.objective is not None:
            return self.objective

        assert self._preset is not None  # enforced by __post_init__
        p = self._preset
        return _build_preset_objective(
            objective_type=p.objective_type,
            risk_aversion=p.risk_aversion,
            alpha=p.alpha,
            horizons=horizons,
            n_scenarios=n_scenarios,
            transaction_cost=p.transaction_cost,
            transaction_cost_weight=p.transaction_cost_weight,
            holding_cost=p.holding_cost,
            holding_cost_weight=p.holding_cost_weight,
        )

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
        objective = self._resolve_objective(horizons=horizons, n_scenarios=n_scenarios)
        return MultiPeriodOptimizer(
            objective=objective,
            horizons=horizons,
            n_assets=n_assets,
            n_scenarios=n_scenarios,
            constraints=self.constraints,
            allow_borrow=self.allow_borrow,
        )

    def solve(
        self,
        *,
        moments: HorizonMoments,
        current_weights: NDArray[np.floating],
        current_cash: float | None = None,
        inputs: dict[str, Any] | None = None,
        **solver_options: Any,
    ) -> MPOResult:
        """
        Compile and solve the problem for the given moments and portfolio state.

        All size parameters (``horizons``, ``n_assets``, ``n_scenarios``) are
        derived directly from ``moments`` — callers do not need to pass them
        separately.

        Parameters
        ----------
        moments:
            Horizon moments produced by :class:`HorizonMoments`.  Determines
            the problem size and supplies all data parameters to the solver.
        current_weights:
            Current risky-asset weights, shape ``(n_assets,)``.
        current_cash:
            Current cash weight.  If ``None``, inferred as
            ``1 - sum(current_weights)``.
        inputs:
            Optional extra inputs forwarded to objective handlers (e.g.
            ``{"prices": ..., "volume": ...}`` for the transaction-cost term).
        **solver_options:
            Forwarded verbatim to the underlying CVXPY solver, merged with
            any options stored on the problem.
        """
        optimizer = self.compile(
            horizons=moments.n_horizons,
            n_assets=moments.n_assets,
            n_scenarios=moments.n_scenarios,
        )
        options = {**self.solver_options, **solver_options}

        logger.debug(
            "Dispatching MPO solve: horizons=%d n_assets=%d n_scenarios=%d cutting_plane=%s",
            moments.n_horizons,
            moments.n_assets,
            moments.n_scenarios,
            optimizer.uses_cutting_plane,
        )

        if optimizer.uses_cutting_plane:
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
