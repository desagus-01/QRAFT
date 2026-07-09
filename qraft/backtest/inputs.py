from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from qraft.backtest.engine.schedule import DecisionPoint, decision_points
from qraft.construction.inputs import PolicyInputRequirements, build_policy_input_table
from qraft.construction.optimization.inputs import InputPlan
from qraft.construction.optimization.inputs import PolicyInputs
from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.core.snapshot import (
    MarketSnapshot,
)
from qraft.forecast.forecaster import Forecaster, ForecastSource
from qraft.utils.log import info_event

logger = logging.getLogger(__name__)


@runtime_checkable
class PolicyInputsProvider(Protocol):
    """Maps a decision snapshot to the PolicyInputs the policy optimises against."""

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs: ...


@dataclass
class DateCache:
    inner: PolicyInputsProvider
    _table: dict[datetime, PolicyInputs] = field(default_factory=dict, init=False)
    _universe: tuple[str, ...] | None = field(default=None, init=False)

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
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
    """Serve PolicyInputs from a ``{date: PolicyInputs}`` table (BYO moments)."""

    table: Mapping[datetime, PolicyInputs]

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
        inputs = self.table.get(snapshot.t)
        if inputs is None:
            raise KeyError(f"No precomputed PolicyInputs for {snapshot.t!r}")
        return inputs


def precompute_inputs(
    market: MarketData,
    schedule: RebalanceSchedule,
    warmup: int,
    *,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
    source: ForecastSource
    | PolicyInputsProvider
    | dict[datetime, PolicyInputs]
    | None = None,
    policy: PolicyInputRequirements | None = None,
    dtype: type = np.float64,
    step_size: int = 1,
) -> dict[datetime, PolicyInputs]:
    """Build and freeze policy inputs for the exact backtest decision schedule."""
    points = decision_points(market, schedule, warmup, step_size=step_size)
    return precompute_inputs_for_points(
        points,
        market,
        forecaster=forecaster,
        plan=plan,
        source=source,
        policy=policy,
        dtype=dtype,
    )


def precompute_inputs_for_points(
    points: list[DecisionPoint],
    market: MarketData,
    *,
    forecaster: Forecaster | None = None,
    plan: InputPlan | None = None,
    source: ForecastSource
    | PolicyInputsProvider
    | dict[datetime, PolicyInputs]
    | None = None,
    policy: PolicyInputRequirements | None = None,
    dtype: type = np.float64,
) -> dict[datetime, PolicyInputs]:
    """Build policy inputs from already-materialized decision snapshots."""
    if isinstance(source, dict):
        table = source
    elif isinstance(source, PolicyInputsProvider):
        table = {
            point.decision_bar: source.for_date(point.snapshot, point.index)
            for point in points
        }
    else:
        forecast_source = source if source is not None else forecaster
        if forecast_source is None:
            raise TypeError("precompute_inputs requires source or forecaster")
        if plan is None:
            raise TypeError("forecast-backed precompute_inputs requires plan")
        table = build_policy_input_table(
            (point.snapshot for point in points),
            forecast_source,
            plan=plan,
            policy=policy,
            dtype=dtype,
            market=market,
        )
    info_event(
        logger,
        "policy_inputs.completed",
        "Policy inputs precomputed",
        decisions=len(table),
        source_type=type(source).__name__
        if source is not None
        else type(forecaster).__name__,
    )
    return table
