from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol, cast, runtime_checkable

import numpy as np

import logging

from qraft.construction.optimization.inputs import (
    AssetDiagnostics,
    InputPlan,
    PolicyInputs,
    RequiredPolicyInputs,
)
from qraft.core.market import MarketData
from qraft.core.snapshot import MarketSnapshot, forecast_snapshot_from_decision_snapshot
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.forecast.forecaster import Forecaster, ForecastSource
from qraft.forecast.run import (
    ForecastRecipeHistory,
    ForecastRun,
    build_forecast_recipe_history,
    build_forecast_recipe_history_from_snapshots,
    simulate_forecast_paths_from_snapshots,
)
from qraft.utils.log import info_event


logger = logging.getLogger(__name__)


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
    market: MarketData | None = None,
) -> dict[datetime, PolicyInputs]:
    """Build ``{date: PolicyInputs}`` from decision snapshots and a forecast source."""
    market_snapshots = list(snapshots)
    info_event(
        logger,
        "policy_inputs.started",
        "Building policy input table",
        decisions=len(market_snapshots),
        expected_returns=plan.expected_returns,
        risk=policy_risk_source(plan, policy),
        max_horizons=plan.max_horizons,
        dtype=getattr(dtype, "__name__", str(dtype)),
    )
    if policy is not None:
        required = policy.required_inputs()
        if (
            not required.covariances
            and not required.scenarios
            and plan.expected_returns != "forecast"
        ):
            info_event(
                logger,
                "policy_inputs.completed",
                "Policy inputs not required by policy",
                decisions=0,
                expected_returns=plan.expected_returns,
                risk=policy_risk_source(plan, policy),
            )
            return {}

    forecasts = forecast_run_for_source(
        market_snapshots, forecast_source, market=market
    )
    forecast_paths = (
        [step.forecast for step in forecasts.steps]
        if isinstance(forecasts, ForecastRun)
        else list(forecasts)
    )
    if isinstance(forecasts, ForecastRun):
        forecast_diagnostics = [step.diagnostics for step in forecasts.steps]
        forecast_invariance_drops = [step.invariance_drops for step in forecasts.steps]
    else:
        forecast_diagnostics = [None] * len(forecast_paths)
        forecast_invariance_drops = [()] * len(forecast_paths)

    table: dict[datetime, PolicyInputs] = {}
    for snapshot, forecast, diagnostics, invariance_drops in zip(
        market_snapshots,
        forecast_paths,
        forecast_diagnostics,
        forecast_invariance_drops,
        strict=True,
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
            cash_return=snapshot.cash_rate,
            asset_diagnostics=asset_diagnostics,
            invariance_drops=invariance_drops,
        )
        if dtype != np.float64:
            inputs = inputs.astype(dtype)
        table[snapshot.t] = inputs

    first_date = min(table) if table else None
    last_date = max(table) if table else None
    n_assets = len(next(iter(table.values())).assets) if table else 0
    info_event(
        logger,
        "policy_inputs.completed",
        "Policy input table built",
        decisions=len(table),
        assets=n_assets,
        first_date=first_date,
        last_date=last_date,
        expected_returns=plan.expected_returns,
        risk=policy_risk_source(plan, policy),
    )

    return table


def forecast_run_for_source(
    market_snapshots: list[MarketSnapshot],
    forecast_source: ForecastSource,
    *,
    market: MarketData | None = None,
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

    if not market_snapshots:
        return ForecastRun(
            recipe_history=ForecastRecipeHistory(
                periods=(),
                pipeline_config=forecast_source.pipeline,
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
            refit_every=forecast_source.refit_every,
            reselect_on_universe_change=forecast_source.reselect_on_universe_change,
            seed=forecast_source.seed,
            pipeline_config=forecast_source.pipeline,
        )
    else:
        min_history = min(
            snapshot.history.values.height for snapshot in market_snapshots
        )
        recipe_history = build_forecast_recipe_history(
            market,
            min_history=min_history,
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
