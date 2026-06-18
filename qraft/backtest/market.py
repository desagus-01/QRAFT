from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from qraft.core.panel import ScenarioPanel


@dataclass(frozen=True, slots=True)
class Market:
    t: datetime
    t_next: datetime
    assets: list[str]
    history: ScenarioPanel
    current_prices: NDArray[np.floating]

    current_prices: NDArray[np.floating]
    cash_return: float
