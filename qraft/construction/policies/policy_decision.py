from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    asset_order: list[str]
    target_weights_risk: NDArray[np.floating]
    target_cash_weight: float
    cash_return: NDArray[np.floating] | None = None
    diagnostics: Any | None = None
    hold: bool = False
