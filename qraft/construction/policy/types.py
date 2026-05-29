from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from construction.state import PortfolioState
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PolicyInputs:
    step: int


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    asset_order: list[str]
    target_weights: NDArray[np.floating]
    target_cash_weight: float
    diagnostics: Any | None = None

    @property
    def target_weights_risk_dict(self) -> dict[str, NDArray[np.floating]]:
        return dict(zip(self.asset_order, self.target_weights))

    @property
    def total_target_weights(self) -> NDArray[np.floating]:
        return np.append(self.target_weights, self.target_cash_weight)

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
            target_weights=target_weights,
            target_cash_weight=self.target_cash_weight,
            diagnostics=None,
        )
