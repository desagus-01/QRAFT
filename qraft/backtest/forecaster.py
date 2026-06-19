from dataclasses import dataclass
from typing import Protocol

from qraft.backtest.market import MarketSnapshot
from qraft.core.configs import PipelineConfig, SimulationForecastConfig
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths
from qraft.forecast.pipelines.forecasting import run_forecast


class Forecaster(Protocol):
    def forecast(
        self, snapshot: MarketSnapshot, universe: AssetUniverse
    ) -> ForecastPaths: ...


@dataclass(frozen=True, slots=True)
class PipelineForecaster:
    simulation_config: SimulationForecastConfig
    pipeline_config: PipelineConfig | None = None
    seed: int | None = None

    def forecast(
        self, snapshot: MarketSnapshot, universe: AssetUniverse
    ) -> ForecastPaths:
        return run_forecast(
            panel=snapshot.history,
            universe=universe,
            seed=self.seed,
            simulation_config=self.simulation_config,
            pipeline_config=self.pipeline_config,
        )
