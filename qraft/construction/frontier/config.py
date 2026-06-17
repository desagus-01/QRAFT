from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from qraft.construction.optimization.constraints import PortfolioConstraint
from qraft.construction.optimization.objectives.specs import (
    HoldingCost,
    TransactionCost,
)
from qraft.construction.optimization.presets import PreMadeObjectives


class FrontierKind(str, Enum):
    EX_ANTE_MPO = "ex_ante_mpo"


def _neutral_holding_cost() -> HoldingCost:
    return HoldingCost(short_fees=0.0, long_fees=0.0, dividends=0.0)


RiskMeasure = Literal["volatility", "cvar_in_model", "cvar_terminal"]


@dataclass(frozen=True, slots=True)
class MPOFrontierConfig:
    risk_measure: RiskMeasure
    risk_aversions: Sequence[float]
    transaction_cost_weight: float = 1.0
    objective_type: PreMadeObjectives = "mean_covariance"
    alpha: float | None = 0.05
    constraints: Sequence[PortfolioConstraint] = ()
    allow_borrow: bool = False
    max_iter: int = 200
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    transaction_cost: TransactionCost | None = field(default_factory=TransactionCost)
    holding_cost: HoldingCost | None = field(default_factory=_neutral_holding_cost)
    holding_cost_weight: float = 1.0
    tail_metric_alpha: float = 0.05
    cvar_mode: Literal["none", "ex_post"] = "ex_post"

    def __post_init__(self) -> None:
        _validate_grid("risk_aversions", self.risk_aversions)

        if self.transaction_cost_weight < 0 or not np.isfinite(
            self.transaction_cost_weight
        ):
            raise ValueError("transaction_cost_weight must be finite and non-negative.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if not 0.0 < self.tail_metric_alpha <= 1.0:
            raise ValueError("tail_metric_alpha must be in (0, 1].")
        if self.holding_cost_weight < 0 or not np.isfinite(self.holding_cost_weight):
            raise ValueError("holding_cost_weight must be finite and non-negative.")
        if self.risk_measure.startswith("cvar") and self.cvar_mode == "none":
            raise ValueError(
                f"risk measure {self.risk_measure} requires a dvar_mode input"
            )


def _validate_grid(name: str, values: Sequence[float]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value.")
    for value in values:
        if value < 0 or not np.isfinite(value):
            raise ValueError(f"{name} values must be finite and non-negative.")
