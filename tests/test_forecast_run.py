from datetime import datetime

import polars as pl

from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.core.universe import AssetUniverse
from qraft.forecast.run import (
    ForecastRecipeHistory,
    RecipePeriod,
    build_forecast_recipe_history,
    simulate_forecast_paths,
)


def _forecast_paths(universe: AssetUniverse | None = None):
    import numpy as np

    from qraft.forecast.forecast_paths import ForecastPaths

    if universe is None:
        universe = AssetUniverse.factors_free(["A"])
    return ForecastPaths(
        asset_paths={"A": np.array([[15.0], [16.0]])},
        dates=pl.Series("date", [datetime(2024, 1, 7)]),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 14.0},
        universe=universe,
    )


def _market(universe: AssetUniverse | None = None) -> MarketData:
    if universe is None:
        universe = AssetUniverse.factors_free(["A"])
    data = pl.DataFrame(
        {
            "date": [datetime(2024, 1, day) for day in range(1, 7)],
            "A": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "B": [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        }
    )
    return MarketData.from_prices(data.select("date", *universe.all_tickers), universe)


def _patch_runner(monkeypatch):
    selected = []
    applied = []

    def create_forecast_recipe(**kwargs):
        recipe = object()
        selected.append((kwargs["universe"], recipe, kwargs["data"].height))
        return recipe

    def apply_forecast_recipe(recipe, *args, **kwargs):
        applied.append(recipe)
        return object()

    def forecast_from_fit(**kwargs):
        return object()

    monkeypatch.setattr(
        "qraft.forecast.run.create_forecast_recipe", create_forecast_recipe
    )
    monkeypatch.setattr(
        "qraft.forecast.run.apply_forecast_recipe", apply_forecast_recipe
    )
    monkeypatch.setattr("qraft.forecast.run.forecast_from_fit", forecast_from_fit)
    return selected, applied


def test_forecast_recipe_history_records_refit_periods(monkeypatch):
    selected, applied = _patch_runner(monkeypatch)
    history = build_forecast_recipe_history(
        _market(),
        min_history=2,
        refit_every=2,
    )

    assert len(selected) == 2
    assert len(applied) == 0
    assert len(history.periods) == 2
    assert history.periods[0].start == datetime(2024, 1, 2)
    assert history.periods[0].end == datetime(2024, 1, 4)
    assert history.periods[1].end is None
    assert [data_height for _, _, data_height in selected] == [2, 4]


def test_forecast_run_applies_existing_recipe_history(monkeypatch):
    _patch_runner(monkeypatch)
    history = build_forecast_recipe_history(
        _market(AssetUniverse.factors_free(["A", "B"])),
        min_history=3,
        refit_every=12,
    )
    run = simulate_forecast_paths(
        _market(AssetUniverse.factors_free(["A", "B"])),
        history,
        min_history=3,
        forecast_cadence="every_bar",
    )

    assert [step.action for step in run.steps] == [
        "selected_recipe",
        "applied_recipe",
        "applied_recipe",
    ]
    assert len(run.recipe_history.periods) == 1
    assert run.steps[0].as_of == datetime(2024, 1, 3)


def test_forecast_cadence_selects_backtest_style_market_bars(monkeypatch):
    _patch_runner(monkeypatch)
    history = build_forecast_recipe_history(
        _market(),
        min_history=2,
        refit_every=1,
    )
    run = simulate_forecast_paths(
        _market(),
        history,
        min_history=2,
        forecast_cadence="every_bar",
    )

    assert [step.as_of for step in run.steps] == [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
        datetime(2024, 1, 4),
        datetime(2024, 1, 5),
    ]


def test_forecast_run_passes_seed_to_each_forecast_step(monkeypatch):
    captured: list[int | None] = []

    def forecast_from_fit(**kwargs):
        captured.append(kwargs["seed"])
        return object()

    _patch_runner(monkeypatch)
    monkeypatch.setattr("qraft.forecast.run.forecast_from_fit", forecast_from_fit)
    history = build_forecast_recipe_history(
        _market(),
        min_history=2,
        refit_every=12,
    )

    simulate_forecast_paths(
        _market(),
        history,
        min_history=2,
        forecast_cadence="every_bar",
        seed=7,
    )

    assert captured == [7, 8, 9, 10]


def test_recipe_history_period_at_reports_available_bounds():
    recipe = object()
    history = ForecastRecipeHistory(
        periods=(
            RecipePeriod(
                start=datetime(2024, 1, 3),
                end=None,
                recipe=recipe,
                universe=("A",),
                selected_at_step=0,
            ),
        ),
        pipeline_config=object(),
    )

    try:
        history.period_at(datetime(2024, 1, 2))
    except KeyError as exc:
        assert "recipe history starts at" in str(exc)
    else:
        raise AssertionError("period_at should reject dates before the first period")


def test_forecast_package_does_not_import_backtest_or_construction():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in (root / "qraft" / "forecast").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "qraft.core.schedule":
                    continue
                assert not node.module.startswith("qraft.backtest")
                assert not node.module.startswith("qraft.construction")


def test_policy_input_precompute_refits_recipes_on_market_bars(monkeypatch):
    from qraft.backtest.simulator import precompute_inputs
    from qraft.construction.optimization.inputs import InputPlan
    from qraft.forecast.forecaster import Forecaster

    selected_steps: list[int] = []

    def create_forecast_recipe(**kwargs):
        selected_steps.append(kwargs["data"].height)
        return object()

    def apply_forecast_recipe(*args, **kwargs):
        return object()

    def forecast_from_fit(**kwargs):
        return _forecast_paths()

    monkeypatch.setattr(
        "qraft.forecast.run.create_forecast_recipe", create_forecast_recipe
    )
    monkeypatch.setattr(
        "qraft.forecast.run.apply_forecast_recipe", apply_forecast_recipe
    )
    monkeypatch.setattr("qraft.forecast.run.forecast_from_fit", forecast_from_fit)

    market = _market()
    table = precompute_inputs(
        market,
        schedule=RebalanceSchedule("every_bar"),
        warmup=2,
        source=Forecaster(refit_every=2),
        plan=InputPlan(expected_returns="forecast", risk="cvar"),
    )

    assert selected_steps == [2, 4]
    assert len(table) == 4
