from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Mapping

from qraft.core.schedule import Cadence, RebalanceSchedule
from qraft.core.configs import (
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.core.snapshot import ForecastSnapshot, forecast_snapshot_at
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.forecast.pipelines.fitted_universe import (
    ForecastRecipe,
    apply_forecast_recipe,
    create_forecast_recipe,
    fit_diagnostics,
    mark_frequent_variance_cap_binding,
)
from qraft.forecast.pipelines.forecasting import forecast_from_fit


@dataclass(frozen=True, slots=True)
class RecipePeriod:
    start: datetime
    end: datetime | None
    recipe: ForecastRecipe
    universe: tuple[str, ...]
    selected_at_step: int
    diagnostics: Mapping[str, object] | None = None
    invariance_drops: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class ForecastRecipeHistory:
    periods: tuple[RecipePeriod, ...]
    pipeline_config: PipelineConfig

    def period_at(self, as_of: datetime) -> tuple[int, RecipePeriod]:
        if not self.periods:
            raise KeyError(f"No forecast recipes are available for {as_of!r}")
        for index, period in enumerate(self.periods):
            if period.start <= as_of and (period.end is None or as_of < period.end):
                return index, period
        first = self.periods[0].start
        last = self.periods[-1].end
        raise KeyError(
            f"No forecast recipe active at {as_of!r}; recipe history starts at "
            f"{first!r} and ends at {last!r}"
        )


@dataclass(frozen=True, slots=True)
class ForecastStep:
    as_of: datetime
    recipe_period_index: int
    forecast: ForecastPaths
    action: Literal["selected_recipe", "applied_recipe"]
    diagnostics: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ForecastRun:
    recipe_history: ForecastRecipeHistory
    steps: tuple[ForecastStep, ...]


def build_forecast_recipe_history(
    market,
    *,
    min_history: int,
    refit_every: int,
    reselect_on_universe_change: bool = True,
    seed: int | None = None,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
) -> ForecastRecipeHistory:
    """Build recipes from every-bar snapshots; ``refit_every`` is in bars."""
    if min_history < 1 or refit_every < 1:
        raise ValueError("min_history and refit_every must be >= 1")
    snapshots = _forecast_snapshots_from_market(
        market,
        min_history=min_history,
        cadence="every_bar",
    )
    return build_forecast_recipe_history_from_snapshots(
        snapshots,
        refit_every=refit_every,
        reselect_on_universe_change=reselect_on_universe_change,
        seed=seed,
        pipeline_config=pipeline_config,
    )


def simulate_forecast_paths(
    market,
    recipe_history: ForecastRecipeHistory,
    *,
    min_history: int,
    forecast_cadence: Cadence = "quarter_end",
    seed: int | None = None,
    simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
) -> ForecastRun:
    if min_history < 1:
        raise ValueError("min_history must be >= 1")
    snapshots = _forecast_snapshots_from_market(
        market,
        min_history=min_history,
        cadence=forecast_cadence,
    )
    return simulate_forecast_paths_from_snapshots(
        snapshots,
        recipe_history,
        pipeline_config=recipe_history.pipeline_config,
        seed=seed,
        simulation_config=simulation_config,
    )


def build_forecast_recipe_history_from_snapshots(
    snapshots: Iterable[ForecastSnapshot],
    *,
    refit_every: int,
    reselect_on_universe_change: bool = True,
    seed: int | None = None,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
) -> ForecastRecipeHistory:
    """Build recipes from supplied snapshots; ``refit_every`` is in snapshot steps."""
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    recipe: ForecastRecipe | None = None
    last_universe: tuple[str, ...] | None = None
    current_period: RecipePeriod | None = None
    periods: list[RecipePeriod] = []

    for step, snapshot in enumerate(snapshots):
        universe_key = tuple(snapshot.universe.all_tickers)
        universe_changed = universe_key != last_universe
        rebuild_recipe = (
            recipe is None
            or (reselect_on_universe_change and universe_changed)
            or step % refit_every == 0
        )
        if not rebuild_recipe:
            last_universe = universe_key
            continue

        panel = snapshot.history
        recipe = create_forecast_recipe(
            data=panel.to_frame(),
            prob=panel.prob,
            universe=snapshot.universe,
            pipeline_config=pipeline_config,
            seed=_seed_for(seed, step),
        )
        if current_period is not None:
            periods[-1] = RecipePeriod(
                start=current_period.start,
                end=snapshot.as_of,
                recipe=current_period.recipe,
                universe=current_period.universe,
                selected_at_step=current_period.selected_at_step,
                diagnostics=current_period.diagnostics,
                invariance_drops=current_period.invariance_drops,
            )
        current_period = RecipePeriod(
            start=snapshot.as_of,
            end=None,
            recipe=recipe,
            universe=universe_key,
            selected_at_step=step,
            invariance_drops=recipe.invariance_drops,
        )
        periods.append(current_period)
        last_universe = universe_key

    return ForecastRecipeHistory(tuple(periods), pipeline_config=pipeline_config)


def simulate_forecast_paths_from_snapshots(
    snapshots: Iterable[ForecastSnapshot],
    recipe_history: ForecastRecipeHistory,
    *,
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    seed: int | None = None,
    simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
) -> ForecastRun:
    steps: list[ForecastStep] = []

    for step, snapshot in enumerate(snapshots):
        period_index, period = recipe_history.period_at(snapshot.as_of)
        panel = snapshot.history
        data = panel.to_frame()
        fit = apply_forecast_recipe(period.recipe, data, panel.prob, pipeline_config)
        forecast = forecast_from_fit(
            panel=panel,
            asset_universe=snapshot.universe,
            universe_fit=fit,
            last_data=data.tail(1),
            n_rows=data.height,
            seed=_seed_for(seed, step),
            simulation_config=simulation_config,
        )
        if getattr(fit, "simulation_forecasts", None):
            diagnostics = mark_frequent_variance_cap_binding(
                fit_diagnostics(fit),
                pipeline_config.volatility.variance_cap_frequent_bind_threshold,
            )
        else:
            diagnostics = None
        steps.append(
            ForecastStep(
                as_of=snapshot.as_of,
                recipe_period_index=period_index,
                forecast=forecast,
                action=(
                    "selected_recipe"
                    if period.start == snapshot.as_of
                    else "applied_recipe"
                ),
                diagnostics=diagnostics,
                invariance_drops=period.invariance_drops,
            )
        )

    return ForecastRun(recipe_history=recipe_history, steps=tuple(steps))


_build_forecast_recipe_history_from_snapshots = (
    build_forecast_recipe_history_from_snapshots
)
_build_forecast_run_from_snapshots = simulate_forecast_paths_from_snapshots


def _forecast_snapshots_from_market(
    market,
    *,
    min_history: int,
    cadence: Cadence,
) -> list[ForecastSnapshot]:
    snapshots: list[ForecastSnapshot] = []
    all_bars = market.trading_bars
    forecast_bars = {d for d, _ in RebalanceSchedule(cadence).decision_steps(all_bars)}

    for i, bar in enumerate(all_bars):
        if i + 1 < min_history or bar not in forecast_bars:
            continue
        if i + 1 >= len(all_bars):
            continue
        snapshots.append(forecast_snapshot_at(market, bar))
    return snapshots


def _seed_for(seed: int | None, step: int) -> int | None:
    return None if seed is None else seed + step
