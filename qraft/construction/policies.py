from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from construction.optimization.constraints import PortfolioConstraint
from construction.optimization.moments import HorizonMoments
from construction.optimization.objectives.specs import HoldingCost, TransactionCost
from construction.optimization.optimization import MPOResult
from construction.optimization.presets import PreMadeObjectives
from construction.optimization.problem import MPOProblem
from construction.state import PortfolioState
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    asset_order: list[str]
    target_weights_risk: NDArray[np.floating]
    target_cash_weight: float
    diagnostics: Any | None = None

    @property
    def target_weights_risk_dict(self) -> dict[str, NDArray[np.floating]]:
        return dict(zip(self.asset_order, self.target_weights_risk))

    @property
    def total_target_weights(self) -> NDArray[np.floating]:
        return np.append(self.target_weights_risk, self.target_cash_weight)

    @property
    def total_target_weights_dict(self) -> dict[str, NDArray[np.floating]]:
        return dict(zip(self.asset_order + ["cash"], self.total_target_weights))


class PolicyProtocol(Protocol):
    name: str

    def decide(
        self, state: PortfolioState, moments: HorizonMoments
    ) -> PolicyDecision: ...


def _decision_from_mpo(result: MPOResult) -> PolicyDecision:
    return PolicyDecision(
        asset_order=result.assets,
        target_weights_risk=result.target_weights,
        target_cash_weight=result.target_cash,
        diagnostics=result,
    )


@dataclass(frozen=True, slots=True)
class EqualWeightPolicy:
    """Allocate equally across all risky assets, holding a fixed cash weight."""

    target_cash_weight: float
    name: str = "equal_weight"

    def decide(self, state: PortfolioState, moments: HorizonMoments) -> PolicyDecision:
        risky_weights = 1.0 - self.target_cash_weight
        n_assets = len(state.asset_order)
        target_weights = np.full(n_assets, risky_weights / n_assets)
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=target_weights,
            target_cash_weight=self.target_cash_weight,
            diagnostics=None,
        )


@dataclass(frozen=True, slots=True)
class MPOPolicy:
    problem: MPOProblem
    name: str = "mpo"

    @classmethod
    def preset(
        cls,
        objective_type: PreMadeObjectives,
        risk_aversion: float,
        *,
        name: str = "mpo",
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
        """
        Create an MPOPolicy from a named preset objective.
        Examples
        --------
        >>> policy = MPOPolicy.preset("mean_covariance", risk_aversion=2.0)
        >>> policy = MPOPolicy.preset(
        ...     "cvar_auto",
        ...     risk_aversion=3.0,
        ...     constraints=(LongOnly(), TurnoverLimit(0.10)),
        ... )
        """
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
        )

    def decide(self, state: PortfolioState, moments: HorizonMoments) -> PolicyDecision:
        result = self.problem.solve(
            moments=moments,
            current_weights=state.asset_weights,
            current_cash=float(state.cash_weight),
        )
        return _decision_from_mpo(result)
