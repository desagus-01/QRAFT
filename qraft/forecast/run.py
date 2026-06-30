from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

from qraft.core.policy_inputs import (
    PolicyInputConfig,
    PolicyInputs,
    RequiredPolicyInputs,
)
from qraft.core.configs import (
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.core.panel import ScenarioPanel
from qraft.core.snapshot import ForecastSnapshot
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.forecast.pipelines.fitted_universe import (
    ForecastRecipe,
    apply_recipe,
    select_recipe,
)
from qraft.forecast.pipelines.forecasting import _forecast_from_fit


@runtime_checkable
class PolicyInputRequirements(Protocol):
    def required_inputs(self) -> RequiredPolicyInputs: ...


@dataclass(frozen=True, slots=True)
class ForecastCadencePolicy:
    refit_every: int
    forecast_every: int
    reselect_on_universe_change: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.refit_every < 1 or self.forecast_every < 1:
            raise ValueError("refit_every and forecast_every must be >= 1")
        if self.forecast_every > self.refit_every:
            raise ValueError("forecast_every must be <= refit_every")


@dataclass(frozen=True, slots=True)
class RecipePeriod:
    start: datetime
    end: datetime | None
    recipe: ForecastRecipe
    universe: tuple[str, ...]
    selected_at_step: int
    diagnostics: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ForecastStep:
    as_of: datetime
    recipe_period_index: int
    forecast: ForecastPaths | None
    policy_inputs: PolicyInputs
    action: Literal["selected_recipe", "applied_recipe", "reused_inputs"]


@dataclass(frozen=True, slots=True)
class ForecastRun:
    recipe_schedule: tuple[RecipePeriod, ...]
    steps: tuple[ForecastStep, ...]

    def policy_inputs_table(self) -> dict[datetime, PolicyInputs]:
        return {step.as_of: step.policy_inputs for step in self.steps}


def build_forecast_run(
    snapshots: Iterable[ForecastSnapshot],
    *,
    cadence_policy: ForecastCadencePolicy,
    input_config: PolicyInputConfig,
    policy: PolicyInputRequirements | None = None,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
    dtype: type = np.float64,
) -> ForecastRun:
    recipe: ForecastRecipe | None = None
    cached_inputs: PolicyInputs | None = None
    last_universe: tuple[str, ...] | None = None
    current_period: RecipePeriod | None = None
    periods: list[RecipePeriod] = []
    steps: list[ForecastStep] = []

    for step, snapshot in enumerate(snapshots):
        universe_key = tuple(snapshot.universe.all_tickers)
        universe_changed = universe_key != last_universe
        rebuild_recipe = (
            recipe is None
            or (cadence_policy.reselect_on_universe_change and universe_changed)
            or step % cadence_policy.refit_every == 0
        )
        reforecast = (
            rebuild_recipe
            or cached_inputs is None
            or step % cadence_policy.forecast_every == 0
        )

        if not reforecast:
            assert cached_inputs is not None
            assert current_period is not None
            steps.append(
                ForecastStep(
                    as_of=snapshot.as_of,
                    recipe_period_index=len(periods) - 1,
                    forecast=None,
                    policy_inputs=cached_inputs,
                    action="reused_inputs",
                )
            )
            continue

        panel = snapshot.history
        data = panel.to_frame()
        prob = panel.prob
        seed = _seed_for(cadence_policy.seed, step)

        if rebuild_recipe:
            if current_period is not None:
                periods[-1] = RecipePeriod(
                    start=current_period.start,
                    end=snapshot.as_of,
                    recipe=current_period.recipe,
                    universe=current_period.universe,
                    selected_at_step=current_period.selected_at_step,
                    diagnostics=current_period.diagnostics,
                )
            recipe = select_recipe(
                data=data,
                prob=prob,
                universe=snapshot.universe,
                pipeline_config=pipeline_config,
                seed=seed,
            )
            current_period = RecipePeriod(
                start=snapshot.as_of,
                end=None,
                recipe=recipe,
                universe=universe_key,
                selected_at_step=step,
            )
            periods.append(current_period)
            last_universe = universe_key
            action: Literal["selected_recipe", "applied_recipe"] = "selected_recipe"
        else:
            assert recipe is not None
            assert current_period is not None
            action = "applied_recipe"

        assert recipe is not None
        fit = apply_recipe(recipe, data, prob, pipeline_config)
        forecast = _forecast_from_fit(
            panel=panel,
            asset_universe=snapshot.universe,
            universe_fit=fit,
            last_data=data.tail(1),
            n_rows=data.height,
            seed=seed,
            simulation_config=simulation_config,
        )
        inputs = _build_inputs(
            forecast,
            input_config=input_config,
            policy=policy,
            history=panel,
            as_of=snapshot.as_of,
            cash_return=snapshot.cash_rate,
        )
        if dtype != np.float64:
            inputs = inputs.astype(dtype)
        cached_inputs = inputs
        steps.append(
            ForecastStep(
                as_of=snapshot.as_of,
                recipe_period_index=len(periods) - 1,
                forecast=forecast,
                policy_inputs=inputs,
                action=action,
            )
        )

    return ForecastRun(recipe_schedule=tuple(periods), steps=tuple(steps))


def _risk_source(
    input_config: PolicyInputConfig, policy: PolicyInputRequirements | None
):
    if input_config.risk is not None:
        _validate_explicit_risk(input_config.risk, policy)
        return input_config.risk
    if policy is None:
        return "both"
    return policy.required_inputs().risk_source


def _validate_explicit_risk(risk, policy: PolicyInputRequirements | None) -> None:
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
            f"input_config.risk={risk!r} does not satisfy policy requirements: "
            f"missing {', '.join(missing)}. Omit risk to infer it from the "
            "policy, or use risk='both'."
        )


def _build_inputs(
    forecasts: ForecastPaths,
    *,
    input_config: PolicyInputConfig,
    policy: PolicyInputRequirements | None,
    history: ScenarioPanel | None = None,
    as_of: datetime | None = None,
    cash_return: float | None = None,
) -> PolicyInputs:
    cfg = input_config
    return PolicyInputs.from_policy_sources(
        forecasts=forecasts,
        cash_path=cfg.cash_path,
        expected_returns=cfg.expected_returns,
        risk=_risk_source(input_config, policy),
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


def _seed_for(seed: int | None, step: int) -> int | None:
    return None if seed is None else seed + step
