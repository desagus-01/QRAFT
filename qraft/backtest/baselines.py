from dataclasses import dataclass
from typing import Any

import numpy as np

from qraft.construction.optimization.moments import PolicyInputs
from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState


@dataclass(frozen=True, slots=True)
class AllCashPolicy:
    name: str = "all_cash"
    min_history: int = 0

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: PolicyInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=np.zeros(len(state.asset_order)),
            target_cash_weight=1.0,
        )


@dataclass(frozen=True, slots=True)
class NoTradePolicy:
    name: str = "no_trade"
    min_history: int = 0

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: PolicyInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=state.asset_weights,
            target_cash_weight=float(state.cash_weight),
        )
