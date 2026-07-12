from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from qraft.backtest.engine.schedule import DecisionPoint, decision_points
from qraft.construction.inputs import (
    OptimizerInputRequirements,
    build_policy_input_table,
)
from qraft.construction.optimization.inputs import InputPlan
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


@dataclass
class DateCache:
    inner: OptimizerInputsProvider
    _table: dict[datetime, OptimizerInputs] = field(default_factory=dict, init=False)
    _universe: tuple[str, ...] | None = field(default=None, init=False)

    def for_date(self, snapshot: MarketSnapshot, step: int) -> OptimizerInputs:
        universe = tuple(snapshot.universe.all_tickers)
        if self._universe is None:
            self._universe = universe
        elif universe != self._universe:
            raise ValueError(
                "DateCache requires a fixed universe; the asset set changed "
                "mid-run. Use a fresh provider per universe."
            )
        cached = self._table.get(snapshot.t)
        if cached is None:
            cached = self.inner.for_date(snapshot, step)
            self._table[snapshot.t] = cached
        return cached


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
    schedule: RebalanceSchedule,
    warmup: int,
    *,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
    forecasts: ForecastSpec
    | OptimizerInputsProvider
    | dict[datetime, OptimizerInputs]
    | None = None,
    policy: OptimizerInputRequirements | None = None,
    dtype: type = np.float64,
    step_size: int = 1,
) -> dict[datetime, OptimizerInputs]:
    """Build and freeze optimizer inputs for the exact backtest decision schedule."""
    points = decision_points(market, schedule, warmup, step_size=step_size)
    return precompute_inputs_for_points(
        points,
        market,
        forecaster=forecaster,
        plan=plan,
        forecasts=forecasts,
        policy=policy,
        dtype=dtype,
    )


def precompute_inputs_for_points(
    points: list[DecisionPoint],
    market: MarketData,
    *,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
    forecasts: ForecastSpec
    | OptimizerInputsProvider
    | dict[datetime, OptimizerInputs]
    | None = None,
    policy: OptimizerInputRequirements | None = None,
    dtype: type = np.float64,
) -> dict[datetime, OptimizerInputs]:
    """Build optimizer inputs from already-materialized decision snapshots."""
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
        if plan is None:
            raise TypeError("forecast-backed precompute_inputs requires plan")
        table = build_policy_input_table(
            (point.snapshot for point in points),
            forecast_input,
            plan=plan,
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
