from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from qraft.core.panel import ScenarioPanel
from qraft.core.scenarios.views import ViewedDistribution
from qraft.core.universe import AssetUniverse


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    """Causal forecast view at as_of."""

    as_of: datetime
    universe: AssetUniverse
    history: ScenarioPanel
    cash_rate: float | None = None
    applied_views: ViewedDistribution | None = None


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Decision view at t -- only information available at or before t."""

    t: datetime
    t_next: datetime
    universe: AssetUniverse
    history: ScenarioPanel
    prices_t: NDArray[np.floating]
    cash_rate: float
    applied_views: ViewedDistribution | None = None


def forecast_snapshot_at(market, t: datetime) -> ForecastSnapshot:
    if hasattr(market, "forecast_snapshot_at"):
        return market.forecast_snapshot_at(t)
    return ForecastSnapshot(
        as_of=t,
        universe=market.universe,
        history=market.history_through(t),
        cash_rate=market.cash_rate_asof(t),
        applied_views=market.applied_views_at(t),
    )


def forecast_snapshot_from_decision_snapshot(snapshot) -> ForecastSnapshot:
    return ForecastSnapshot(
        as_of=snapshot.t,
        universe=snapshot.universe,
        history=snapshot.history,
        cash_rate=snapshot.cash_rate,
        applied_views=snapshot.applied_views,
    )
