from datetime import datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from qraft.core.configs import (
    PipelineConfig,
    PreprocessConfig,
    SimulationForecastConfig,
)
from qraft.core.panel import ScenarioPanel
from qraft.core.universe import AssetUniverse
from qraft.forecast.forecast_paths import InnovationPaths
from qraft.forecast.pipelines import forecasting
from qraft.forecast.pipelines.fitted_universe import ForecastRecipe


def test_innovation_paths_rejects_invalid_path_probs() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        InnovationPaths(
            values=np.ones((2, 3, 1)),
            path_probs=np.array([0.5, 0.4]),
        )


class DummyFittedUniverse:
    def __init__(self) -> None:
        self.invariants = ScenarioPanel(
            values=pl.DataFrame({"asset": [0.01, 0.02]}),
            dates=pl.Series("date", [datetime(2024, 1, 2), datetime(2024, 1, 3)]),
            prob=np.full(2, 0.5),
            kind="invariant",
        )
        self.inverse_specs = {"asset": []}
        self.preprocess = SimpleNamespace(assets_already_iid=["asset"])
        self.assets = ["asset"]
        self.models = {
            "asset": SimpleNamespace(
                volatility_res=SimpleNamespace(
                    admissible=False,
                    fallback_reason="garch_refit_failed",
                ),
                quality=SimpleNamespace(
                    grade="C",
                    score=55.0,
                    reason_codes=("VOL_FALLBACK_BEST_IC_NO_DIAG_PASS",),
                ),
            )
        }
        self.simulation_forecasts = {
            "asset": SimpleNamespace(variance_cap_diagnostics={"bind_rate": 0.25})
        }

    def simulate(self, innovation_paths):
        return {
            "asset": np.ones((innovation_paths.shape[0], innovation_paths.shape[1]))
        }

    def recipe(self):
        return ForecastRecipe(
            detrend={},
            deseason={},
            needs_modelling=["asset"],
            mean_orders={"asset": (1, 0)},
            vol_orders={"asset": (1, 0, 1)},
            mean_fallback_identity={"asset": None},
            vol_distributions={"asset": "normal"},
            quality={"asset": self.models["asset"].quality},
            admissible={"asset": False},
            fallback_reason={"asset": "garch_refit_failed"},
            variance_cap_diagnostics={"asset": {"bind_rate": 0.25}},
            survivors=["asset"],
        )


def _panel() -> ScenarioPanel:
    return ScenarioPanel.from_log_prices(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "asset": [1.0, 1.1],
            }
        ),
        prob=np.full(2, 0.5),
    )


def test_run_forecast_passes_top_level_seed_to_stochastic_steps(monkeypatch) -> None:
    captured: dict[str, int | None] = {}

    def fake_create_forecast_recipe(**kwargs):
        captured["fit_seed"] = kwargs["seed"]
        return object()

    def fake_apply_forecast_recipe(*args, **kwargs):
        return DummyFittedUniverse()

    def fake_draw_innovations(**kwargs):
        captured["draw_seed"] = kwargs["seed"]
        return InnovationPaths(
            values=np.ones((2, 3, 1)),
            path_probs=np.full(2, 0.5),
        )

    monkeypatch.setattr(
        forecasting, "create_forecast_recipe", fake_create_forecast_recipe
    )
    monkeypatch.setattr(
        forecasting, "apply_forecast_recipe", fake_apply_forecast_recipe
    )
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

    def fake_create_forecast_recipe(**kwargs):
        captured["fit_seed"] = kwargs["seed"]
        return object()

    def fake_apply_forecast_recipe(*args, **kwargs):
        return DummyFittedUniverse()

    def fake_draw_innovations(**kwargs):
        captured["draw_seed"] = kwargs["seed"]
        return InnovationPaths(
            values=np.ones((2, 3, 1)),
            path_probs=np.full(2, 0.5),
        )

    monkeypatch.setattr(
        forecasting, "create_forecast_recipe", fake_create_forecast_recipe
    )
    monkeypatch.setattr(
        forecasting, "apply_forecast_recipe", fake_apply_forecast_recipe
    )
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
    panel = ScenarioPanel.from_levels(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "asset": [100.0, 101.0],
            }
        )
    )

    with pytest.raises(ValueError, match="kind='log_price'"):
        forecasting.run_forecast(
            panel=panel,
            universe=AssetUniverse.factors_free(["asset"]),
        )


def test_draw_innovations_rejects_non_invariant_panel() -> None:
    panel = ScenarioPanel.from_returns(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "asset": [0.01, -0.02],
            }
        )
    )

    with pytest.raises(ValueError, match="kind='invariant'"):
        forecasting.draw_innovations(
            invariants=panel,
            horizon=1,
            n_sims=2,
            seed=1,
        )
