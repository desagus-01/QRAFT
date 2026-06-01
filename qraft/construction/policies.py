from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from construction.optimization.constraints import PortfolioConstraint
from construction.optimization.moments import HorizonMoments
from construction.optimization.optimization import MPOResult
from construction.optimization.presets import PreMadeObjectives
from construction.optimization.problem import MPOProblem, build_preset_problem
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

    def decide(self, state: PortfolioState, moments: HorizonMoments) -> PolicyDecision:
        pass


@dataclass(frozen=True, slots=True)
class EqualWeightPolicy:
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


def _decision_from_mpo(result: MPOResult) -> PolicyDecision:
    return PolicyDecision(
        asset_order=result.assets,
        target_weights_risk=result.target_weights,
        target_cash_weight=result.target_cash,
        diagnostics=result,
    )


@dataclass(frozen=True, slots=True)
class CustomMPOPolicy:
    problem: MPOProblem
    name: str = "custom_mpo"

    def decide(self, state: PortfolioState, moments: HorizonMoments) -> PolicyDecision:
        result = self.problem.solve(
            horizons=moments.n_horizons,
            n_assets=moments.n_assets,
            n_scenarios=moments.scenario_returns.shape[0],
            moments=moments,
            current_weights=state.asset_weights,
            current_cash=float(state.cash_weight),
        )
        return _decision_from_mpo(result)


@dataclass(frozen=True, slots=True)
class MPOPolicy:
    name: PreMadeObjectives
    risk_aversion: float
    cvar_alpha: float | None = 0.05
    constraints: Sequence[PortfolioConstraint] = ()
    allow_borrow: bool = False
    max_iter: int = 200

    def decide(
        self, state: PortfolioState, moments: HorizonMoments, **solver_options
    ) -> PolicyDecision:
        problem = build_preset_problem(
            objective_type=self.name,
            risk_aversion=self.risk_aversion,
            alpha=self.cvar_alpha,
            horizons=moments.n_horizons,
            n_scenarios=moments.scenario_returns.shape[0],
            constraints=self.constraints,
            allow_borrow=self.allow_borrow,
            max_iter=self.max_iter,
            **solver_options,
        )
        return CustomMPOPolicy(problem=problem, name=self.name).decide(state, moments)
