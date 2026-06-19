from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    round_shares: bool = True
    minimum_trade_value: float = 0.0
    max_volume_fraction: float | None = None


@dataclass(frozen=True)
class ExecutionResult:
    requested_weight_trades: NDArray
    requested_share_trades: NDArray
    executed_share_trades: NDArray
    executed_trade_values: NDArray

    shares_after: NDArray
    cash_after_trades: float

    transaction_cost: float
    holding_cost: float
    cash_after_costs: float
