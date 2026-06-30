from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qraft.core.panel import ScenarioPanel
from qraft.core.universe import AssetUniverse


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    """Causal forecast view at as_of."""

    as_of: datetime
    universe: AssetUniverse
    history: ScenarioPanel
    cash_rate: float | None = None
