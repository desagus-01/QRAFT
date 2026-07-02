from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol, cast, runtime_checkable

import numpy as np

from qraft.construction.optimization.inputs import (
    AssetDiagnostics,
    InputPlan,
    PolicyInputs,
    RequiredPolicyInputs,
)
from qraft.core.snapshot import MarketSnapshot, forecast_snapshot_from_decision_snapshot
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.forecast.forecaster import Forecaster, ForecastSource
from qraft.forecast.run import (
    ForecastRecipeHistory,
    ForecastRun,
    build_forecast_recipe_history_from_snapshots,
    simulate_forecast_paths_from_snapshots,
)


@runtime_checkable
class PolicyInputRequirements(Protocol):
    def required_inputs(self) -> RequiredPolicyInputs: ...


def policy_risk_source(
    plan: InputPlan,
    policy: PolicyInputRequirements | None = None,
):
    if plan.risk is not None:
        validate_policy_risk_source(plan.risk, policy)
        return plan.risk
    if policy is None:
        return "both"
    return policy.required_inputs().risk_source


def validate_policy_risk_source(risk, policy: PolicyInputRequirements | None) -> None:
    if policy is None:
        return
    required = policy.required_inputs()
    has_covariance = risk in {"covariance", "both"}
    has_scenarios = risk in {"cvar", "both"}
    missing: list[str] = []
    if required.covariances and not has_covariance:
        missing.append("covariances")
    if required.scenarios and not has_scenarios:
        missing.append("scenario_returns")
    if missing:
        raise ValueError(
            f"plan.risk={risk!r} does not satisfy policy requirements: "
            f"missing {', '.join(missing)}. Omit risk to infer it from the "
            "policy, or use risk='both'."
        )


def build_policy_input_table(
    snapshots: Iterable[MarketSnapshot],
    forecast_source: ForecastSource,
    *,
    plan: InputPlan,
    policy=None,
    dtype: type = np.float64,
) -> dict[datetime, PolicyInputs]:
    """Build ``{date: PolicyInputs}`` from decision snapshots and a forecast source."""
    market_snapshots = list(snapshots)
    if policy is not None:
        required = policy.required_inputs()
        if (
            not required.covariances
            and not required.scenarios
            and plan.expected_returns != "forecast"
        ):
            return {}

    forecasts = forecast_run_for_source(market_snapshots, forecast_source)
    forecast_paths = (
        [step.forecast for step in forecasts.steps]
        if isinstance(forecasts, ForecastRun)
        else list(forecasts)
    )
    forecast_diagnostics = (
        [step.diagnostics for step in forecasts.steps]
        if isinstance(forecasts, ForecastRun)
        else [None] * len(forecast_paths)
    )

    table: dict[datetime, PolicyInputs] = {}
    for snapshot, forecast, diagnostics in zip(
        market_snapshots, forecast_paths, forecast_diagnostics, strict=True
    ):
        asset_diagnostics = ()
        if diagnostics is not None:
            asset_diagnostics = tuple(
                AssetDiagnostics(asset=asset, values=cast(Mapping[str, Any], values))
                for asset, values in diagnostics.items()
            )
        inputs = PolicyInputs.from_policy_sources(
            forecasts=forecast,
            expected_returns=plan.expected_returns,
            risk=policy_risk_source(plan, policy),
            history=snapshot.history,
            max_horizons=plan.max_horizons,
            subset=plan.subset,
            pnl_type=plan.pnl_type,
            expectation_tolerance=plan.expectation_tolerance,
            mean_decay=plan.mean_decay,
            as_of=snapshot.t,
            cash_return=snapshot.cash_rate,
            asset_diagnostics=asset_diagnostics,
        )
        if dtype != np.float64:
            inputs = inputs.astype(dtype)
        table[snapshot.t] = inputs

    return table


def forecast_run_for_source(
    market_snapshots: list[MarketSnapshot],
    forecast_source: ForecastSource,
) -> ForecastRun | Iterable[ForecastPaths]:
    if isinstance(forecast_source, ForecastRun):
        return forecast_source
    if isinstance(forecast_source, ForecastRecipeHistory):
        forecast_snapshots = [
            forecast_snapshot_from_decision_snapshot(snapshot)
            for snapshot in market_snapshots
        ]
        return simulate_forecast_paths_from_snapshots(
            forecast_snapshots,
            forecast_source,
            pipeline_config=forecast_source.pipeline_config,
        )
    if not isinstance(forecast_source, Forecaster):
        return forecast_source

    forecast_snapshots = [
        forecast_snapshot_from_decision_snapshot(snapshot)
        for snapshot in market_snapshots
    ]
    recipe_history = build_forecast_recipe_history_from_snapshots(
        forecast_snapshots,
        refit_every=forecast_source.refit_every,
        reselect_on_universe_change=forecast_source.reselect_on_universe_change,
        seed=forecast_source.seed,
        pipeline_config=forecast_source.pipeline,
    )
    return simulate_forecast_paths_from_snapshots(
        forecast_snapshots,
        recipe_history,
        pipeline_config=forecast_source.pipeline,
        seed=forecast_source.seed,
        simulation_config=forecast_source.simulation,
    )


_forecast_run_for_source = forecast_run_for_source
