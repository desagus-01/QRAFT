from datetime import datetime

import numpy as np
import polars as pl

from qraft.construction.optimization.moments import PolicyInputConfig, PolicyInputs
from qraft.core.panel import ScenarioPanel
from qraft.core.snapshot import ForecastSnapshot
from qraft.core.universe import AssetUniverse
from qraft.forecast.run import ForecastCadencePolicy, build_forecast_run


def _snapshot(day: int, universe: AssetUniverse | None = None) -> ForecastSnapshot:
    t = datetime(2024, 1, day)
    if universe is None:
        universe = AssetUniverse.factors_free(["A"])
    history = ScenarioPanel.from_prices(
        pl.DataFrame({"date": [datetime(2024, 1, 1), t], "A": [9.0, 10.0]})
    )
    return ForecastSnapshot(as_of=t, universe=universe, history=history, cash_rate=0.01)


def _patch_runner(monkeypatch):
    selected = []
    applied = []

    def select_recipe(**kwargs):
        recipe = object()
        selected.append((kwargs["universe"], recipe))
        return recipe

    def apply_recipe(recipe, *args, **kwargs):
        applied.append(recipe)
        return object()

    def forecast_from_fit(**kwargs):
        return object()

    def from_policy_sources(**kwargs):
        return PolicyInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=kwargs["cash_return"]
        )

    monkeypatch.setattr("qraft.forecast.run.select_recipe", select_recipe)
    monkeypatch.setattr("qraft.forecast.run.apply_recipe", apply_recipe)
    monkeypatch.setattr("qraft.forecast.run._forecast_from_fit", forecast_from_fit)
    monkeypatch.setattr(
        "qraft.forecast.run.PolicyInputs.from_policy_sources", from_policy_sources
    )
    return selected, applied


def test_forecast_run_records_cadence_actions(monkeypatch):
    selected, applied = _patch_runner(monkeypatch)
    run = build_forecast_run(
        [_snapshot(2), _snapshot(3), _snapshot(4), _snapshot(5)],
        cadence_policy=ForecastCadencePolicy(refit_every=3, forecast_every=2),
        input_config=PolicyInputConfig(cash_path="unused.csv"),
    )

    assert [step.action for step in run.steps] == [
        "selected_recipe",
        "reused_inputs",
        "applied_recipe",
        "selected_recipe",
    ]
    assert len(selected) == 2
    assert len(applied) == 3
    assert len(run.recipe_schedule) == 2
    assert run.recipe_schedule[0].start == datetime(2024, 1, 2)
    assert run.recipe_schedule[0].end == datetime(2024, 1, 5)
    assert run.recipe_schedule[1].end is None


def test_forecast_run_reselects_on_universe_change(monkeypatch):
    _patch_runner(monkeypatch)
    run = build_forecast_run(
        [
            _snapshot(2, AssetUniverse.factors_free(["A"])),
            _snapshot(3, AssetUniverse.factors_free(["B"])),
        ],
        cadence_policy=ForecastCadencePolicy(refit_every=12, forecast_every=12),
        input_config=PolicyInputConfig(cash_path="unused.csv"),
    )

    assert [step.action for step in run.steps] == ["selected_recipe", "selected_recipe"]
    assert len(run.recipe_schedule) == 2


def test_forecast_package_does_not_import_backtest_or_construction():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in (root / "qraft" / "forecast").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("qraft.backtest")
                assert not node.module.startswith("qraft.construction")
