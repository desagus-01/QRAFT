from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from qraft.core.panel import ScenarioPanel


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    t: datetime
    t_next: datetime
    assets: list[str]
    history: ScenarioPanel
    prices_t: NDArray[np.floating]
