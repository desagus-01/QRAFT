from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np

from qraft.construction.optimization.moments import PolicyInputConfig, PolicyInputs
from qraft.core.configs import (
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    ForecastProviderConfig,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.core.snapshot import MarketSnapshot, forecast_snapshot_from_decision_snapshot
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.forecast.run import (
    ForecastRun,
    build_forecast_recipe_history_from_snapshots,
    simulate_forecast_paths_from_snapshots,
)


def build_policy_input_table(
    snapshots: Iterable[MarketSnapshot],
    forecasts: ForecastRun | Iterable[ForecastPaths],
    *,
    input_config: PolicyInputConfig,
    risk_source,
    dtype: type = np.float64,
) -> dict[datetime, PolicyInputs]:
    """Build ``{date: PolicyInputs}`` from decision snapshots and forecasts."""
    market_snapshots = list(snapshots)
    forecast_paths = (
        [step.forecast for step in forecasts.steps]
        if isinstance(forecasts, ForecastRun)
        else list(forecasts)
    )

    table: dict[datetime, PolicyInputs] = {}
    for snapshot, forecast in zip(market_snapshots, forecast_paths, strict=True):
        inputs = PolicyInputs.from_policy_sources(
            forecasts=forecast,
            cash_path=input_config.cash_path,
            expected_returns=input_config.expected_returns,
            risk=risk_source,
            history=snapshot.history,
            max_horizons=input_config.max_horizons,
            subset=input_config.subset,
            pnl_type=input_config.pnl_type,
            expectation_tolerance=input_config.expectation_tolerance,
            mean_decay=input_config.mean_decay,
            step_size=input_config.step_size,
            periods_per_year=input_config.periods_per_year,
            as_of=snapshot.t,
            cash_return=snapshot.cash_rate,
        )
        if dtype != np.float64:
            inputs = inputs.astype(dtype)
        table[snapshot.t] = inputs

    return table


def build_policy_input_table_from_forecast_run(
    snapshots: Iterable[MarketSnapshot],
    forecast_run: ForecastRun,
    *,
    input_config: PolicyInputConfig,
    risk_source,
    dtype: type = np.float64,
) -> dict[datetime, PolicyInputs]:
    """Build ``{date: PolicyInputs}`` from a pre-simulated forecast run."""
    return build_policy_input_table(
        snapshots,
        forecast_run,
        input_config=input_config,
        risk_source=risk_source,
        dtype=dtype,
    )


def forecast_policy_input_table(
    snapshots: Iterable[MarketSnapshot],
    *,
    input_config: PolicyInputConfig,
    risk_source,
    provider_config: ForecastProviderConfig,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
    dtype: type = np.float64,
) -> dict[datetime, PolicyInputs]:
    """Forecast decision snapshots, then build ``{date: PolicyInputs}``."""
    market_snapshots = list(snapshots)
    forecast_snapshots = [
        forecast_snapshot_from_decision_snapshot(snapshot)
        for snapshot in market_snapshots
    ]
    recipe_history = build_forecast_recipe_history_from_snapshots(
        forecast_snapshots,
        refit_every=provider_config.refit_every,
        reselect_on_universe_change=provider_config.reselect_on_universe_change,
        seed=provider_config.seed,
        pipeline_config=pipeline_config,
    )
    run = simulate_forecast_paths_from_snapshots(
        forecast_snapshots,
        recipe_history,
        pipeline_config=pipeline_config,
        seed=provider_config.seed,
        simulation_config=simulation_config,
    )
    return build_policy_input_table(
        market_snapshots,
        run,
        input_config=input_config,
        risk_source=risk_source,
        dtype=dtype,
    )
