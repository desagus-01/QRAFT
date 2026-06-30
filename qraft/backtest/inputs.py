from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

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
from qraft.core.snapshot import (
    ForecastSnapshot,
    MarketSnapshot,
    forecast_snapshot_from_decision_snapshot,
)
from qraft.forecast.run import (
    ForecastRun,
    _build_forecast_recipe_history_from_snapshots,
    _build_forecast_run_from_snapshots,
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
        market_snapshots = list(snapshots)
        forecast_snapshots = [
            forecast_snapshot_from_decision_snapshot(snapshot)
            for snapshot in market_snapshots
        ]
        run = self.run(forecast_snapshots)
        table: dict[datetime, PolicyInputs] = {}

        for snapshot, step in zip(market_snapshots, run.steps, strict=True):
            inputs = PolicyInputs.from_policy_sources(
                forecasts=step.forecast,
                cash_path=self.input_config.cash_path,
                expected_returns=self.input_config.expected_returns,
                risk=self._risk_source(),
                history=snapshot.history,
                max_horizons=self.input_config.max_horizons,
                subset=self.input_config.subset,
                pnl_type=self.input_config.pnl_type,
                expectation_tolerance=self.input_config.expectation_tolerance,
                mean_decay=self.input_config.mean_decay,
                step_size=self.input_config.step_size,
                periods_per_year=self.input_config.periods_per_year,
                as_of=snapshot.t,
                cash_return=snapshot.cash_rate,
            )
            if self.dtype != np.float64:
                inputs = inputs.astype(self.dtype)
            table[snapshot.t] = inputs

        return table

    def run(self, snapshots: Iterable[ForecastSnapshot]) -> ForecastRun:
        snapshots = list(snapshots)
        recipe_history = _build_forecast_recipe_history_from_snapshots(
            snapshots,
            refit_every=self.provider_config.refit_every,
            reselect_on_universe_change=self.provider_config.reselect_on_universe_change,
            seed=self.provider_config.seed,
            pipeline_config=self.pipeline_config,
        )
        return _build_forecast_run_from_snapshots(
            snapshots,
            recipe_history,
            pipeline_config=self.pipeline_config,
            simulation_config=self.simulation_config,
        )

    def _risk_source(self):
        if self.input_config.risk is not None:
            self._validate_explicit_risk(self.input_config.risk)
            return self.input_config.risk
        if self.policy is None:
            return "both"
        return self.policy.required_inputs().risk_source

    def _validate_explicit_risk(self, risk) -> None:
        if self.policy is None:
            return
        required = self.policy.required_inputs()
        has_covariance = risk in {"covariance", "both"}
        has_scenarios = risk in {"cvar", "both"}
        missing: list[str] = []
        if required.covariances and not has_covariance:
            missing.append("covariances")
        if required.scenarios and not has_scenarios:
            missing.append("scenario_returns")
        if missing:
            raise ValueError(
                f"input_config.risk={risk!r} does not satisfy policy requirements: "
                f"missing {', '.join(missing)}. Omit risk to infer it from the "
                "policy, or use risk='both'."
            )


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
