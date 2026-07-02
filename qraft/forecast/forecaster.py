from __future__ import annotations

from dataclasses import dataclass, field

from qraft.core.configs import PipelineConfig, SimulationForecastConfig
from qraft.core.schedule import Cadence
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths
from qraft.forecast.pipelines.forecasting import run_forecast
from qraft.forecast.run import (
    ForecastRecipeHistory,
    ForecastRun,
    build_forecast_recipe_history,
    simulate_forecast_paths,
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

    def forecast(self, panel, universe: AssetUniverse) -> ForecastPaths:
        return run_forecast(
            panel,
            universe,
            seed=self.seed,
            pipeline_config=self.pipeline,
            simulation_config=self.simulation,
        )

    def recipes(self, market, min_history: int) -> ForecastRecipeHistory:
        return build_forecast_recipe_history(
            market,
            min_history=min_history,
            refit_every=self.refit_every,
            reselect_on_universe_change=self.reselect_on_universe_change,
            seed=self.seed,
            pipeline_config=self.pipeline,
        )

    def run(
        self,
        market,
        min_history: int,
        cadence: Cadence,
    ) -> ForecastRun:
        return simulate_forecast_paths(
            market,
            self.recipes(market, min_history),
            min_history=min_history,
            forecast_cadence=cadence,
            seed=self.seed,
            simulation_config=self.simulation,
        )
