from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from qraft.construction.market_snapshot import MarketSnapshot
from qraft.construction.optimization.moments import (
    PolicyInputConfig,
    PolicyInputs,
    RequiredPolicyInputs,
    RiskSource,
)
from qraft.core.configs import (
    DEFAULT_FORECAST_PROVIDER_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    ForecastProviderConfig,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.core.panel import ScenarioPanel
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths
from qraft.forecast.pipelines.fitted_universe import (
    FittedUniverse,
    ForecastRecipe,
    recondition,
)
from qraft.forecast.pipelines.forecasting import _forecast_from_fit

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

        self._recipe: ForecastRecipe | None = None
        self._cached_inputs: PolicyInputs | None = None
        self._last_universe: tuple[str, ...] | None = None

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
        universe_key = tuple(snapshot.universe.all_tickers)
        universe_changed = universe_key != self._last_universe

        rebuild_recipe = (
            self._recipe is None
            or universe_changed
            or step % self.provider_config.refit_every == 0
        )
        reforecast = (
            rebuild_recipe
            or self._cached_inputs is None
            or step % self.provider_config.forecast_every == 0
        )
        if not reforecast:
            assert self._cached_inputs is not None
            return self._cached_inputs

        panel = snapshot.history
        universe = snapshot.universe
        data = panel.to_frame()
        prob = panel.prob
        seed = self._seed_for(step)

        if rebuild_recipe:
            fit = FittedUniverse.fit(
                data=data,
                prob=prob,
                universe=universe,
                pipeline_config=self.pipeline_config,
                seed=seed,
            )
            self._recipe = fit.recipe()
            self._last_universe = universe_key
            logger.info("step %d: rebuilt forecast recipe (full fit)", step)
        else:
            assert self._recipe is not None
            fit = recondition(self._recipe, data, prob, self.pipeline_config)
            logger.info("step %d: reconditioned forecast (cached recipe)", step)

        forecasts = self._forecast(panel, universe, fit, data, seed)
        inputs = self._build_inputs(
            forecasts,
            history=panel,
            as_of=snapshot.t,
            cash_return=snapshot.cash_rate,
        )
        if self.dtype != np.float64:
            inputs = inputs.astype(self.dtype)
        self._cached_inputs = inputs
        return inputs

    def _forecast(
        self, panel, universe: AssetUniverse, fit: FittedUniverse, data, seed
    ) -> ForecastPaths:
        return _forecast_from_fit(
            panel=panel,
            asset_universe=universe,
            universe_fit=fit,
            last_data=data.tail(1),
            n_rows=data.height,
            seed=seed,
            simulation_config=self.simulation_config,
        )

    def _build_inputs(
        self,
        forecasts: ForecastPaths,
        history: ScenarioPanel | None = None,
        *,
        as_of: datetime | None = None,
        cash_return: float | None = None,
    ) -> PolicyInputs:
        cfg = self.input_config
        risk = self._risk_source()
        return PolicyInputs.from_policy_sources(
            forecasts=forecasts,
            cash_path=cfg.cash_path,
            expected_returns=cfg.expected_returns,
            risk=risk,
            history=history,
            max_horizons=cfg.max_horizons,
            subset=cfg.subset,
            pnl_type=cfg.pnl_type,
            expectation_tolerance=cfg.expectation_tolerance,
            mean_decay=cfg.mean_decay,
            step_size=cfg.step_size,
            periods_per_year=cfg.periods_per_year,
            as_of=as_of,
            cash_return=cash_return,
        )

    def _risk_source(self) -> RiskSource:
        if self.input_config.risk is not None:
            self._validate_explicit_risk(self.input_config.risk)
            return self.input_config.risk
        if self.policy is None:
            return "both"
        return self.policy.required_inputs().risk_source

    def _validate_explicit_risk(self, risk: RiskSource) -> None:
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

    def _seed_for(self, step: int) -> int | None:
        seed = self.provider_config.seed
        return None if seed is None else seed + step


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
