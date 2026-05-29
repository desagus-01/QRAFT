from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from construction.optimization.moments import HorizonMoments
from construction.optimization.optimization import PreMadeObjectives
from construction.optimization.pre_built import multi_period_optimization
from construction.state import PortfolioState
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PolicyInputs:
    step: int
    moments: HorizonMoments


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

    def decide(self, state: PortfolioState, inputs: PolicyInputs) -> PolicyDecision:
        pass


@dataclass(frozen=True, slots=True)
class EqualWeightPolicy:
    target_cash_weight: float
    name: str = "equal_weight"

    def decide(self, state: PortfolioState, inputs: PolicyInputs) -> PolicyDecision:
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
    name: PreMadeObjectives
    risk_aversion: float

    def decide(self, state: PortfolioState, inputs: PolicyInputs) -> PolicyDecision:
        return multi_period_optimization(
            objective_type=self.name,
            step=inputs.step,
            n_assets=inputs.moments.n_assets,
            risk_aversion=self.risk_aversion,
            moments=inputs.moments,
            current_weights=state.asset_weights,
            current_cash=float(state.cash_weight),
        )
