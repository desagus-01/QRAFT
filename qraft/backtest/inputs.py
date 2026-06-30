from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

from qraft.construction.market_snapshot import (
    MarketSnapshot,
    forecast_snapshot_from_market,
)
from qraft.construction.optimization.moments import (
    PolicyInputConfig,
    PolicyInputs,
    RequiredPolicyInputs,
)
from qraft.core.configs import (
    DEFAULT_FORECAST_PROVIDER_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    ForecastProviderConfig,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.core.snapshot import ForecastSnapshot
from qraft.forecast.run import (
    ForecastCadencePolicy,
    ForecastRun,
    _risk_source,
    build_forecast_run,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class PolicyInputsProvider(Protocol):
    """Maps a decision snapshot to the PolicyInputs the policy optimises against."""

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs: ...


@runtime_checkable
class PolicyInputRequirements(Protocol):
    def required_inputs(self) -> RequiredPolicyInputs: ...


class ForecastInputsProvider:
    """Forecast-then-build-moments, with a forecast cadence."""

    def __init__(
        self,
        input_config: PolicyInputConfig,
        *,
        policy: PolicyInputRequirements | None = None,
        provider_config: ForecastProviderConfig = DEFAULT_FORECAST_PROVIDER_CONFIG,
        pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
        simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
        dtype: type = np.float64,
    ) -> None:
        self.input_config = input_config
        self.policy = policy
        self.provider_config = provider_config
        self.pipeline_config = pipeline_config
        self.simulation_config = simulation_config
        self.dtype = dtype

        self._snapshots: list[MarketSnapshot] = []
        self._table: dict[datetime, PolicyInputs] = {}

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
        if snapshot.t not in self._table:
            self._snapshots.append(snapshot)
            self._table = self.build(self._snapshots)
        return self._table[snapshot.t]

    def build(self, snapshots) -> dict[datetime, PolicyInputs]:
        return self.run(
            forecast_snapshot_from_market(snapshot) for snapshot in snapshots
        ).policy_inputs_table()

    def run(self, snapshots: Iterable[ForecastSnapshot]) -> ForecastRun:
        cadence_policy = ForecastCadencePolicy(
            refit_every=self.provider_config.refit_every,
            forecast_every=self.provider_config.forecast_every,
            seed=self.provider_config.seed,
        )
        return build_forecast_run(
            snapshots,
            cadence_policy=cadence_policy,
            input_config=self.input_config,
            policy=self.policy,
            pipeline_config=self.pipeline_config,
            simulation_config=self.simulation_config,
            dtype=self.dtype,
        )

    def _risk_source(self):
        return _risk_source(self.input_config, self.policy)


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
