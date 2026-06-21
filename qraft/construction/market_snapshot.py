from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from qraft.core.panel import ScenarioPanel
from qraft.forecast.forecast_paths import AssetUniverse


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Decision view at t — only information available at or before t."""

    t: datetime
    t_next: datetime
    universe: AssetUniverse
    history: ScenarioPanel
    prices_t: NDArray[np.floating]
    cash_rate: float
