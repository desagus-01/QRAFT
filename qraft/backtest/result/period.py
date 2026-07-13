from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import ForecastPaths


@dataclass(frozen=True, slots=True)
class BacktestPeriod:
    decision_bar: datetime
    execution_bar: datetime
    state_before: PortfolioState
    decision: PolicyDecision
    executed_share_trades: NDArray[np.floating]
    state_after: PortfolioState
    solver_status: str | None
    cost: float = 0.0
    forecasts: ForecastPaths | None = None
    decision_error: str | None = None
    dropped_assets: tuple[str, ...] = ()
    asset_diagnostics: tuple[Any, ...] = ()
    invariance_drops: tuple[Any, ...] = ()
    view_diagnostics: Any = None
    optimizer_inputs: OptimizerInputs | None = None
