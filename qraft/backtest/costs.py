from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qraft.construction.optimization.objectives.specs import (
    HoldingCost,
    TransactionCost,
    holding_cost_value,
    transaction_cost_value,
)


@dataclass(frozen=True, slots=True)
class CostModel:
    transaction: TransactionCost | None = None
    holding: HoldingCost | None = None

    @classmethod
    def frictionless(cls) -> "CostModel":
        return cls()

    @classmethod
    def from_policy(cls, policy: object) -> "CostModel":
        get_specs = getattr(policy, "cost_specs", None)
        if not callable(get_specs):
            return cls()
        transaction, holding = get_specs()
        return cls(transaction=transaction, holding=holding)

    def trade_charge(
        self,
        executed_shares: NDArray[np.floating],
        prices: NDArray[np.floating],
        nav: float,
        *,
        sigma: NDArray[np.floating] | None = None,
    ) -> float:
        if self.transaction is None or nav <= 0.0:
            return 0.0
        return transaction_cost_value(
            self.transaction,
            weight_trades=(executed_shares * prices) / nav,
            nav=nav,
            prices=prices,
            sigma=sigma,
        )

    def holding_charge(
        self,
        weights: NDArray[np.floating],
        nav: float,
        n_periods: int = 1,
    ) -> float:
        if self.holding is None or nav <= 0.0:
            return 0.0
        return holding_cost_value(self.holding, weights, nav=nav, n_periods=n_periods)
