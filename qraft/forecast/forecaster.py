from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypeAlias

from qraft.core.configs import PipelineConfig, SimulationForecastConfig
from qraft.core.market import MarketData
from qraft.core.schedule import Cadence
from qraft.core.snapshot import ForecastSnapshot
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.forecast.run import (
    ForecastRecipeHistory,
    ForecastRun,
    build_forecast_recipe_history,
    build_forecast_recipe_history_from_snapshots,
    simulate_forecast_paths,
    simulate_forecast_paths_from_snapshots,
)


@dataclass(frozen=True, slots=True)
class Forecaster:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    simulation: SimulationForecastConfig = field(
        default_factory=SimulationForecastConfig
    )
    refit_every: int = 12
    reselect_on_universe_change: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")

    def recipes(self, market: MarketData, min_history: int) -> ForecastRecipeHistory:
        return build_forecast_recipe_history(
            market,
            min_history=min_history,
            refit_every=self.refit_every,
            reselect_on_universe_change=self.reselect_on_universe_change,
            seed=self.seed,
            pipeline_config=self.pipeline,
        )

    def recipes_from_snapshots(
        self,
        snapshots: Iterable[ForecastSnapshot],
    ) -> ForecastRecipeHistory:
        return build_forecast_recipe_history_from_snapshots(
            snapshots,
            refit_every=self.refit_every,
            reselect_on_universe_change=self.reselect_on_universe_change,
            seed=self.seed,
            pipeline_config=self.pipeline,
        )

    def run(
        self,
        market: MarketData,
        min_history: int,
        cadence: Cadence,
        recipes: ForecastRecipeHistory | None = None,
    ) -> ForecastRun:
        return simulate_forecast_paths(
            market,
            recipes or self.recipes(market, min_history),
            min_history=min_history,
            forecast_cadence=cadence,
            seed=self.seed,
            simulation_config=self.simulation,
        )

    def run_from_snapshots(
        self,
        snapshots: Iterable[ForecastSnapshot],
        recipes: ForecastRecipeHistory | None = None,
    ) -> ForecastRun:
        snapshot_list = list(snapshots)
        recipe_history = recipes or self.recipes_from_snapshots(snapshot_list)
        return simulate_forecast_paths_from_snapshots(
            snapshot_list,
            recipe_history,
            pipeline_config=self.pipeline,
            seed=self.seed,
            simulation_config=self.simulation,
        )


ForecastSpec: TypeAlias = (
    Forecaster | ForecastRun | ForecastRecipeHistory | Iterable[ForecastPaths]
)
