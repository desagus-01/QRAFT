import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from qraft.construction.optimization.constraints import PortfolioConstraint
from qraft.construction.optimization.inputs import (
    OptimizerInputs,
    RequiredOptimizerInputs,
)
from qraft.construction.optimization.objectives.specs import (
    HoldingCost,
    TransactionCost,
)
from qraft.construction.optimization.optimization import (
    MPOFailure,
    MPOResult,
    MultiPeriodOptimizer,
    OptimizationFailure,
)
from qraft.construction.optimization.presets import PreMadeObjectives, PresetCosts
from qraft.construction.optimization.problem import MPOProblem
from qraft.construction.policies.policy_decision import PolicyDecision
from qraft.construction.state import PortfolioState, align_state_to_assets

logger = logging.getLogger(__name__)


class PolicyProtocol(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def min_history(self) -> int: ...

    def decide(
        self,
        state: PortfolioState,
        optimizer_inputs: OptimizerInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision: ...

    def required_inputs(self) -> RequiredOptimizerInputs: ...


def _decision_from_mpo(
    result: MPOResult, cash_return: NDArray[np.floating] | None
) -> PolicyDecision:
    return PolicyDecision(
        asset_order=result.assets,
        target_weights_risk=result.target_weights,
        target_cash_weight=result.target_cash,
        cash_return=cash_return,
        diagnostics=result,
    )


@dataclass(frozen=True, slots=True)
class EqualWeightPolicy:
    target_cash_weight: float
    name: str = "equal_weight"
    min_history: int = 0

    def required_inputs(self) -> RequiredOptimizerInputs:
        return RequiredOptimizerInputs()

    def decide(
        self,
        state: PortfolioState,
        optimizer_inputs: OptimizerInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        risky_weights = 1.0 - self.target_cash_weight
        n_assets = len(state.asset_order)
        target_weights = np.full(n_assets, risky_weights / n_assets)
        cash_return = (
            optimizer_inputs.cash_return
            if isinstance(optimizer_inputs, OptimizerInputs)
            and optimizer_inputs.cash_return is not None
            else np.zeros(1)
        )
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=target_weights,
            target_cash_weight=self.target_cash_weight,
            cash_return=cash_return,
            diagnostics=None,
        )


@dataclass(frozen=True, slots=True)
class MPOPolicy:
    """Multi-period optimizer policy over explicit, pre-built optimizer inputs."""

    problem: MPOProblem
    min_history: int = 0
    name: str = "mpo"
    _optimizer_cache: dict[tuple[tuple[str, ...], int, int], MultiPeriodOptimizer] = (
        field(default_factory=dict, compare=False, repr=False)
    )

    @classmethod
    def preset(
        cls,
        objective_type: PreMadeObjectives,
        risk_aversion: float | None = None,
        *,
        name: str = "mpo",
        min_history: int = 0,
        alpha: float | None = 0.05,
        constraints: Sequence[PortfolioConstraint] = (),
        allow_borrow: bool = False,
        max_iter: int = 200,
        transaction_cost: TransactionCost | None = None,
        transaction_cost_weight: float = 1.0,
        holding_cost: HoldingCost | None = None,
        holding_cost_weight: float = 1.0,
        costs: PresetCosts = "default",
        **solver_options: Any,
    ) -> "MPOPolicy":
        return cls(
            problem=MPOProblem.preset(
                objective_type,
                risk_aversion,
                alpha=alpha,
                constraints=constraints,
                allow_borrow=allow_borrow,
                max_iter=max_iter,
                transaction_cost=transaction_cost,
                transaction_cost_weight=transaction_cost_weight,
                holding_cost=holding_cost,
                holding_cost_weight=holding_cost_weight,
                costs=costs,
                **solver_options,
            ),
            name=name,
            min_history=min_history,
        )

    def cost_specs(self) -> tuple[TransactionCost | None, HoldingCost | None]:
        return self.problem.cost_specs()

    def required_inputs(self) -> RequiredOptimizerInputs:
        return self.problem.required_inputs()

    def _get_optimizer(self, optimizer_inputs: OptimizerInputs) -> MultiPeriodOptimizer:
        key = (
            tuple(optimizer_inputs.assets),
            optimizer_inputs.n_horizons,
            optimizer_inputs.n_scenarios,
        )
        optimizer = self._optimizer_cache.get(key)
        if optimizer is None:
            optimizer = self.problem.compile(
                horizons=optimizer_inputs.n_horizons,
                n_assets=optimizer_inputs.n_assets,
                n_scenarios=optimizer_inputs.n_scenarios,
            )
            self._optimizer_cache[key] = optimizer
        return optimizer

    def decide(
        self,
        state: PortfolioState,
        optimizer_inputs: OptimizerInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        if not isinstance(optimizer_inputs, OptimizerInputs):
            raise ValueError(
                "MPOPolicy requires explicit OptimizerInputs. Call decide() (or "
                "optimize()) with pre-built OptimizerInputs."
            )
        return self.optimize(
            state=state, optimizer_inputs=optimizer_inputs, inputs=inputs
        )

    def optimize(
        self,
        state: PortfolioState,
        optimizer_inputs: OptimizerInputs,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        current_weights, current_cash, dropped, dropped_weight = align_state_to_assets(
            state, optimizer_inputs.assets
        )
        if dropped:
            logger.warning(
                "optimize(): %d asset(s) dropped by OptimizerInputs %s — "
                "their combined weight (%.4f) transferred to cash for reallocation.",
                len(dropped),
                sorted(dropped),
                dropped_weight,
            )

        optimizer = self._get_optimizer(optimizer_inputs)
        result = optimizer.solve_auto(
            optimizer_inputs=optimizer_inputs,
            current_weights=current_weights,
            current_cash=current_cash,
            inputs=inputs,
            max_iter=self.problem.max_iter,
            **self.problem.solver_options,
        )
        if isinstance(result, MPOFailure):
            raise OptimizationFailure(result)

        return _decision_from_mpo(result, cash_return=optimizer_inputs.cash_return)
