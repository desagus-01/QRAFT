import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from qraft.construction.optimization.constraints import PortfolioConstraint
from qraft.construction.optimization.inputs import PolicyInputs, RequiredPolicyInputs
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
from qraft.construction.optimization.presets import PreMadeObjectives
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
        policy_inputs: PolicyInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision: ...

    def required_inputs(self) -> RequiredPolicyInputs: ...


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

    def required_inputs(self) -> RequiredPolicyInputs:
        return RequiredPolicyInputs()

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: PolicyInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        risky_weights = 1.0 - self.target_cash_weight
        n_assets = len(state.asset_order)
        target_weights = np.full(n_assets, risky_weights / n_assets)
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=target_weights,
            target_cash_weight=self.target_cash_weight,
            cash_return=np.zeros(1),
            diagnostics=None,
        )


@dataclass(frozen=True, slots=True)
class MPOPolicy:
    """Multi-period optimizer policy over explicit, pre-built policy inputs."""

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
                **solver_options,
            ),
            name=name,
            min_history=min_history,
        )

    def cost_specs(self) -> tuple[TransactionCost | None, HoldingCost | None]:
        return self.problem.cost_specs()

    def required_inputs(self) -> RequiredPolicyInputs:
        return self.problem.required_inputs()

    def _get_optimizer(self, moments: PolicyInputs) -> MultiPeriodOptimizer:
        key = (tuple(moments.assets), moments.n_horizons, moments.n_scenarios)
        optimizer = self._optimizer_cache.get(key)
        if optimizer is None:
            optimizer = self.problem.compile(
                horizons=moments.n_horizons,
                n_assets=moments.n_assets,
                n_scenarios=moments.n_scenarios,
            )
            self._optimizer_cache[key] = optimizer
        return optimizer

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: PolicyInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        if not isinstance(policy_inputs, PolicyInputs):
            raise ValueError(
                "MPOPolicy requires explicit PolicyInputs. Call decide() (or "
                "optimize()) with pre-built PolicyInputs."
            )
        return self.optimize(state=state, policy_inputs=policy_inputs, inputs=inputs)

    def optimize(
        self,
        state: PortfolioState,
        policy_inputs: PolicyInputs,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        current_weights, current_cash, dropped, dropped_weight = align_state_to_assets(
            state, policy_inputs.assets
        )
        if dropped:
            logger.warning(
                "optimize(): %d asset(s) dropped by PolicyInputs %s — "
                "their combined weight (%.4f) transferred to cash for reallocation.",
                len(dropped),
                sorted(dropped),
                dropped_weight,
            )

        optimizer = self._get_optimizer(policy_inputs)
        result = optimizer.solve_auto(
            moments=policy_inputs,
            current_weights=current_weights,
            current_cash=current_cash,
            inputs=inputs,
            max_iter=self.problem.max_iter,
            **self.problem.solver_options,
        )
        if isinstance(result, MPOFailure):
            raise OptimizationFailure(result)

        return _decision_from_mpo(result, cash_return=policy_inputs.cash_return)
