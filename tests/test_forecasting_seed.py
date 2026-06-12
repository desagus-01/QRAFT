import numpy as np
import polars as pl
import pytest

from qraft.core.configs import (
    PipelineConfig,
    PreprocessConfig,
    SimulationForecastConfig,
)
from qraft.core.panel import ScenarioPanel
from qraft.forecast.forecast_paths import AssetUniverse, InnovationPaths
from qraft.forecast.pipelines import forecasting


class DummyFittedUniverse:
    def __init__(self) -> None:
        self.invariants = ScenarioPanel(
            values=pl.DataFrame({"asset": [0.01, 0.02]}),
            dates=None,
            prob=np.full(2, 0.5),
            kind="invariant",
        )
        self.inverse_specs = {"asset": []}

    def simulate(self, innovation_paths):
        return {
            "asset": np.ones((innovation_paths.shape[0], innovation_paths.shape[1]))
        }


def _panel() -> ScenarioPanel:
    return ScenarioPanel.from_log_prices(
        pl.DataFrame({"asset": [1.0, 1.1]}),
        prob=np.full(2, 0.5),
    )


def test_run_forecast_passes_top_level_seed_to_stochastic_steps(monkeypatch) -> None:
    captured: dict[str, int | None] = {}

    def fake_fit(**kwargs):
        captured["fit_seed"] = kwargs["seed"]
        return DummyFittedUniverse()

    def fake_draw_innovations(**kwargs):
        captured["draw_seed"] = kwargs["seed"]
        return InnovationPaths(
            values=np.ones((2, 3, 1)),
            path_probs=np.full(2, 0.5),
        )

    monkeypatch.setattr(forecasting.FittedUniverse, "fit", fake_fit)
    monkeypatch.setattr(forecasting, "draw_innovations", fake_draw_innovations)

    forecasting.run_forecast(
        panel=_panel(),
        universe=AssetUniverse.factors_free(["asset"]),
        seed=3,
        simulation_config=SimulationForecastConfig(horizon=3, n_sims=2),
        pipeline_config=PipelineConfig(preprocess=PreprocessConfig()),
    )

    assert captured == {
        "fit_seed": 3,
        "draw_seed": 3,
    }


def test_run_forecast_leaves_stochastic_steps_unseeded_when_seed_is_omitted(
    monkeypatch,
) -> None:
    captured: dict[str, int | None] = {}

    def fake_fit(**kwargs):
        captured["fit_seed"] = kwargs["seed"]
        return DummyFittedUniverse()

    def fake_draw_innovations(**kwargs):
        captured["draw_seed"] = kwargs["seed"]
        return InnovationPaths(
            values=np.ones((2, 3, 1)),
            path_probs=np.full(2, 0.5),
        )

    monkeypatch.setattr(forecasting.FittedUniverse, "fit", fake_fit)
    monkeypatch.setattr(forecasting, "draw_innovations", fake_draw_innovations)

    forecasting.run_forecast(
        panel=_panel(),
        universe=AssetUniverse.factors_free(["asset"]),
        simulation_config=SimulationForecastConfig(horizon=3, n_sims=2),
    )

    assert captured == {
        "fit_seed": None,
        "draw_seed": None,
    }


def test_run_forecast_rejects_non_log_price_panel() -> None:
    panel = ScenarioPanel.from_levels(pl.DataFrame({"asset": [100.0, 101.0]}))

    with pytest.raises(ValueError, match="kind='log_price'"):
        forecasting.run_forecast(
            panel=panel,
            universe=AssetUniverse.factors_free(["asset"]),
        )


def test_draw_innovations_rejects_non_invariant_panel() -> None:
    panel = ScenarioPanel.from_returns(pl.DataFrame({"asset": [0.01, -0.02]}))

    with pytest.raises(ValueError, match="kind='invariant'"):
        forecasting.draw_innovations(
            invariants=panel,
            horizon=1,
            n_sims=2,
            seed=1,
        )
