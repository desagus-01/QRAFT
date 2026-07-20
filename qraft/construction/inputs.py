from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol, cast, runtime_checkable

import numpy as np

import logging

from qraft.construction.optimization.inputs import (
    AssetDiagnostics,
    InputPlan,
    OptimizerInputs,
    RequiredOptimizerInputs,
)
from qraft.core.market import MarketData
from qraft.core.snapshot import MarketSnapshot, forecast_snapshot_from_decision_snapshot
from qraft.forecast.forecaster import Forecaster, ForecastSpec
from qraft.forecast.run import (
    ForecastRecipeHistory,
    ForecastRun,
    build_forecast_recipe_history_from_snapshots,
    simulate_forecast_paths_from_snapshots,
)
from qraft.utils.log import info_event


logger = logging.getLogger(__name__)


@runtime_checkable
class OptimizerInputRequirements(Protocol):
    def required_inputs(self) -> RequiredOptimizerInputs: ...


def required_optimizer_inputs(
    policy: OptimizerInputRequirements | None = None,
) -> RequiredOptimizerInputs:
    if policy is None:
        return RequiredOptimizerInputs(mean=True, covariances=True, scenarios=True)
    return policy.required_inputs()


def build_optimizer_input_table(
    snapshots: Iterable[MarketSnapshot],
    forecasts: ForecastSpec,
    *,
    policy=None,
    dtype: type = np.float64,
    market: MarketData | None = None,
) -> dict[datetime, OptimizerInputs]:
    """Build ``{date: OptimizerInputs}`` from decision snapshots and a forecast source."""
    market_snapshots = list(snapshots)
    plan = getattr(policy, "input_plan", InputPlan())
    required = required_optimizer_inputs(policy)
    info_event(
        logger,
        "optimizer_inputs.started",
        "Building optimizer input table",
        decisions=len(market_snapshots),
        expected_returns=plan.expected_returns,
        required_inputs=required.risk_source,
        max_horizons=plan.max_horizons,
        dtype=getattr(dtype, "__name__", str(dtype)),
    )
    if policy is not None:
        if (
            not required.mean
            and not required.covariances
            and not required.scenarios
            and plan.expected_returns != "forecast"
        ):
            info_event(
                logger,
                "optimizer_inputs.completed",
                "Optimizer inputs not required by policy",
                decisions=0,
                expected_returns=plan.expected_returns,
                required_inputs=required.risk_source,
            )
            return {}

    forecasts = forecast_run_for_source(market_snapshots, forecasts, market=market)

    table: dict[datetime, OptimizerInputs] = {}
    for snapshot in market_snapshots:
        step = forecasts.step_at(snapshot.t)
        forecast = step.forecast
        diagnostics = step.diagnostics
        invariance_drops = step.invariance_drops
        asset_diagnostics = ()
        if diagnostics is not None:
            asset_diagnostics = tuple(
                AssetDiagnostics(asset=asset, values=cast(Mapping[str, Any], values))
                for asset, values in diagnostics.items()
            )
        applied_views = getattr(snapshot, "applied_views", None)
        view_diag = applied_views.diagnostics if applied_views is not None else None
        inputs = OptimizerInputs.from_policy_sources(
            forecasts=forecast,
            expected_returns=plan.expected_returns,
            required_inputs=required,
            history=snapshot.history,
            max_horizons=plan.max_horizons,
            subset=plan.subset,
            pnl_type=plan.pnl_type,
            expectation_tolerance=plan.expectation_tolerance,
            mean_decay=plan.mean_decay,
            cash_return=snapshot.cash_rate,
            asset_diagnostics=asset_diagnostics,
            invariance_drops=invariance_drops,
            view_diagnostics=view_diag,
        )
        if dtype != np.float64:
            inputs = inputs.astype(dtype)
        table[snapshot.t] = inputs

    first_date = min(table) if table else None
    last_date = max(table) if table else None
    n_assets = len(next(iter(table.values())).assets) if table else 0
    info_event(
        logger,
        "optimizer_inputs.completed",
        "Optimizer input table built",
        decisions=len(table),
        assets=n_assets,
        first_date=first_date,
        last_date=last_date,
        expected_returns=plan.expected_returns,
        required_inputs=required.risk_source,
    )

    return table


def forecast_run_for_source(
    market_snapshots: list[MarketSnapshot],
    forecasts: ForecastSpec,
    *,
    market: MarketData | None = None,
) -> ForecastRun:
    if isinstance(forecasts, ForecastRun):
        return forecasts
    if isinstance(forecasts, ForecastRecipeHistory):
        forecast_snapshots = [
            forecast_snapshot_from_decision_snapshot(snapshot)
            for snapshot in market_snapshots
        ]
        return simulate_forecast_paths_from_snapshots(
            forecast_snapshots,
            forecasts,
            pipeline_config=forecasts.pipeline_config,
        )
    if not isinstance(forecasts, Forecaster):
        raise TypeError(
            "forecasts must be a Forecaster, ForecastRun, or ForecastRecipeHistory"
        )

    if not market_snapshots:
        return ForecastRun(
            recipe_history=ForecastRecipeHistory(
                periods=(),
                pipeline_config=forecasts.pipeline,
            ),
            steps=(),
        )

    forecast_snapshots = [
        forecast_snapshot_from_decision_snapshot(snapshot)
        for snapshot in market_snapshots
    ]
    if market is None:
        recipe_history = build_forecast_recipe_history_from_snapshots(
            forecast_snapshots,
            new_recipe_every=forecasts.new_recipe_every,
            reselect_on_universe_change=forecasts.reselect_on_universe_change,
            seed=forecasts.seed,
            pipeline_config=forecasts.pipeline,
        )
        return simulate_forecast_paths_from_snapshots(
            forecast_snapshots,
            recipe_history,
            pipeline_config=forecasts.pipeline,
            seed=forecasts.seed,
            simulation_config=forecasts.simulation,
        )
    else:
        min_history = min(
            snapshot.history.values.height for snapshot in market_snapshots
        )
        if (
            len(market_snapshots) == 1
            and market_snapshots[0].t == market.trading_bars[-1]
        ):
            min_history = max(1, min_history - 1)
        recipe_history = forecasts.recipes(market, min_history=min_history)
        return simulate_forecast_paths_from_snapshots(
            forecast_snapshots,
            recipe_history,
            pipeline_config=forecasts.pipeline,
            seed=forecasts.seed,
            simulation_config=forecasts.simulation,
        )


_forecast_run_for_source = forecast_run_for_source
