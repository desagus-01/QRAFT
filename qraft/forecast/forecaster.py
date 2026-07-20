"""High-level forecast facade for recipe selection and path simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from qraft.core.configs import PipelineConfig, SimulationForecastConfig
from qraft.core.market import MarketData
from qraft.core.schedule import Cadence
from qraft.forecast.run import (
    ForecastRecipeHistory,
    ForecastRun,
    build_forecast_recipe_history,
    simulate_forecast_paths,
)


@dataclass(frozen=True, slots=True)
class Forecaster:
    """Configure recipe selection and simulation settings for forecast runs."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    simulation: SimulationForecastConfig = field(
        default_factory=SimulationForecastConfig
    )
    new_recipe_every: int = 12
    new_recipe_cadence: Cadence = "every_bar"
    reselect_on_universe_change: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.new_recipe_every < 1:
            raise ValueError("new_recipe_every must be >= 1")

    def recipes(self, market: MarketData, min_history: int) -> ForecastRecipeHistory:
        """Return a ``ForecastRecipeHistory`` built from market-data snapshots."""
        return build_forecast_recipe_history(
            market,
            min_history=min_history,
            new_recipe_every=self.new_recipe_every,
            new_recipe_cadence=self.new_recipe_cadence,
            reselect_on_universe_change=self.reselect_on_universe_change,
            seed=self.seed,
            pipeline_config=self.pipeline,
        )

    def run(
        self,
        market: MarketData,
        min_history: int,
        forecast_cadence: Cadence,
        recipes: ForecastRecipeHistory | None = None,
    ) -> ForecastRun:
        """Return a ``ForecastRun`` from market data; inspect ``run.steps`` next."""
        return simulate_forecast_paths(
            market,
            recipes or self.recipes(market, min_history),
            min_history=min_history,
            forecast_cadence=forecast_cadence,
            seed=self.seed,
            simulation_config=self.simulation,
        )


ForecastSpec: TypeAlias = Forecaster | ForecastRun | ForecastRecipeHistory
