from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from qraft.core.panel import ScenarioPanel
from qraft.core.snapshot import ForecastSnapshot
from qraft.core.universe import AssetUniverse


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Decision view at t — only information available at or before t."""

    t: datetime
    t_next: datetime
    universe: AssetUniverse
    history: ScenarioPanel
    prices_t: NDArray[np.floating]
    cash_rate: float


def forecast_snapshot_from_market(snapshot: MarketSnapshot) -> ForecastSnapshot:
    return ForecastSnapshot(
        as_of=snapshot.t,
        universe=snapshot.universe,
        history=snapshot.history,
        cash_rate=snapshot.cash_rate,
    )
