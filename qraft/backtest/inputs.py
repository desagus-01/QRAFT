from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from qraft.backtest.engine.schedule import DecisionPoint, decision_points
from qraft.construction.inputs import (
    OptimizerInputRequirements,
    build_optimizer_input_table,
)
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.core.snapshot import (
    MarketSnapshot,
)
from qraft.forecast.forecaster import Forecaster, ForecastSpec
from qraft.utils.log import info_event

logger = logging.getLogger(__name__)


@runtime_checkable
class OptimizerInputsProvider(Protocol):
    """Maps a decision snapshot to the OptimizerInputs the policy optimises against."""

    def for_date(self, snapshot: MarketSnapshot, step: int) -> OptimizerInputs: ...


@dataclass(frozen=True, slots=True)
class PrecomputedInputsProvider:
    """Serve OptimizerInputs from a ``{date: OptimizerInputs}`` table."""

    table: Mapping[datetime, OptimizerInputs]

    def for_date(self, snapshot: MarketSnapshot, step: int) -> OptimizerInputs:
        inputs = self.table.get(snapshot.t)
        if inputs is None:
            raise KeyError(f"No precomputed OptimizerInputs for {snapshot.t!r}")
        return inputs


def precompute_inputs(
    market: MarketData,
    schedule: RebalanceSchedule | None = None,
    warmup: int | None = None,
    *,
    points: list[DecisionPoint] | None = None,
    forecaster: Forecaster | None = None,
    forecasts: ForecastSpec
    | OptimizerInputsProvider
    | dict[datetime, OptimizerInputs]
    | None = None,
    policy: OptimizerInputRequirements | None = None,
    dtype: type = np.float64,
    step_size: int = 1,
) -> dict[datetime, OptimizerInputs]:
    """Build and freeze optimizer inputs for a decision schedule or supplied points."""
    if points is None:
        if schedule is None or warmup is None:
            raise TypeError("precompute_inputs requires points or schedule and warmup")
        points = decision_points(market, schedule, warmup, step_size=step_size)

    if isinstance(forecasts, dict):
        table = forecasts
    elif isinstance(forecasts, OptimizerInputsProvider):
        table = {
            point.decision_bar: forecasts.for_date(point.snapshot, point.index)
            for point in points
        }
    else:
        forecast_input = forecasts if forecasts is not None else forecaster
        if forecast_input is None:
            raise TypeError("precompute_inputs requires forecasts or forecaster")
        table = build_optimizer_input_table(
            (point.snapshot for point in points),
            forecast_input,
            policy=policy,
            dtype=dtype,
            market=market,
        )
    info_event(
        logger,
        "optimizer_inputs.completed",
        "Optimizer inputs precomputed",
        decisions=len(table),
        source_type=type(forecasts).__name__
        if forecasts is not None
        else type(forecaster).__name__,
    )
    return table
