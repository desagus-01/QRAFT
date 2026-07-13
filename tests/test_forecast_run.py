import logging
from datetime import datetime

import polars as pl

from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.core.snapshot import forecast_snapshot_at
from qraft.core.universe import AssetUniverse
from qraft.forecast.forecaster import Forecaster
from qraft.forecast.run import (
    ForecastRecipeHistory,
    RecipePeriod,
    build_forecast_recipe_history,
    simulate_forecast_paths,
    simulate_forecast_paths_from_snapshots,
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
        model_health_frame=pl.DataFrame(
            {
                "asset": ["A"],
                "mean_model": ["(1, 0)"],
                "vol_model": ["(1, 0, 1) normal"],
                "quality_grade": ["C"],
                "quality_score": [55.0],
                "fallback_reason": ["garch_refit_failed"],
                "cap_bind_rate": [0.25],
                "admissible": [False],
            }
        ),
    )


def _market(universe: AssetUniverse | None = None) -> MarketData:
    if universe is None:
        universe = AssetUniverse.factors_free(["A"])
    data = pl.DataFrame(
        {
            "date": [datetime(2024, 1, day) for day in range(1, 7)],
            "A": [18.0, 20.0, 22.0, 24.0, 26.0, 28.0],
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


def test_forecast_run_logs_step_applications_at_debug(monkeypatch, caplog):
    _patch_runner(monkeypatch)
    history = build_forecast_recipe_history(
        _market(AssetUniverse.factors_free(["A", "B"])),
        min_history=3,
        refit_every=12,
    )

    with caplog.at_level(logging.INFO, logger="qraft.forecast.run"):
        simulate_forecast_paths(
            _market(AssetUniverse.factors_free(["A", "B"])),
            history,
            min_history=3,
            forecast_cadence="every_bar",
        )

    events = [record.qraft_event for record in caplog.records]
    assert "forecast.recipe_applied" not in events
    assert "forecast.run_started" in events
    assert "forecast.completed" in events


def test_forecaster_run_returns_forecast_run(monkeypatch):
    _patch_runner(monkeypatch)

    run = Forecaster(refit_every=12).run(
        _market(AssetUniverse.factors_free(["A", "B"])),
        min_history=3,
        cadence="every_bar",
    )

    assert [step.action for step in run.steps] == [
        "selected_recipe",
        "applied_recipe",
        "applied_recipe",
    ]
    assert len(run.recipe_history.periods) == 1


def test_forecaster_run_reuses_supplied_recipes(monkeypatch):
    selected, applied = _patch_runner(monkeypatch)
    market = _market(AssetUniverse.factors_free(["A", "B"]))
    forecaster = Forecaster(refit_every=12)
    recipes = forecaster.recipes(market, min_history=3)
    selected_count = len(selected)
    applied.clear()

    run = forecaster.run(
        market,
        min_history=3,
        cadence="every_bar",
        recipes=recipes,
    )

    assert len(selected) == selected_count
    assert len(applied) == 3
    assert run.recipe_history is recipes


def test_forecaster_run_from_snapshots_reuses_recipes(monkeypatch):
    selected, applied = _patch_runner(monkeypatch)
    market = _market(AssetUniverse.factors_free(["A", "B"]))
    forecaster = Forecaster(refit_every=1)
    recipes = forecaster.recipes(market, min_history=3)
    selected_count = len(selected)
    applied.clear()
    snapshots = [forecast_snapshot_at(market, bar) for bar in market.trading_bars[2:]]

    run = forecaster.run_from_snapshots(snapshots, recipes)

    assert len(selected) == selected_count
    assert len(applied) == 4
    assert run.recipe_history is recipes


def test_forecast_paths_model_health_returns_stored_frame():
    health = _forecast_paths().model_health()

    assert health.columns == [
        "asset",
        "mean_model",
        "vol_model",
        "quality_grade",
        "quality_score",
        "fallback_reason",
        "cap_bind_rate",
        "admissible",
    ]
    assert health.to_dicts() == [
        {
            "asset": "A",
            "mean_model": "(1, 0)",
            "vol_model": "(1, 0, 1) normal",
            "quality_grade": "C",
            "quality_score": 55.0,
            "fallback_reason": "garch_refit_failed",
            "cap_bind_rate": 0.25,
            "admissible": False,
        }
    ]


def test_forecast_run_model_health_stacks_steps_with_recipe_period():
    from qraft.forecast.run import ForecastRun, ForecastStep

    run = ForecastRun(
        recipe_history=ForecastRecipeHistory(periods=(), pipeline_config=object()),
        steps=(
            ForecastStep(
                as_of=datetime(2024, 1, 3),
                recipe_period_index=0,
                forecast=_forecast_paths(),
                action="selected_recipe",
            ),
            ForecastStep(
                as_of=datetime(2024, 1, 4),
                recipe_period_index=0,
                forecast=_forecast_paths(),
                action="applied_recipe",
            ),
        ),
    )

    health = run.model_health()

    assert health.select("asset", "as_of", "recipe_period").to_dicts() == [
        {"asset": "A", "as_of": datetime(2024, 1, 3), "recipe_period": 0},
        {"asset": "A", "as_of": datetime(2024, 1, 4), "recipe_period": 0},
    ]


def test_forecast_paths_plot_asset_paths_returns_figure():
    import matplotlib
    from matplotlib.figure import Figure

    matplotlib.use("Agg")

    fig = _forecast_paths().plot_asset_paths(max_assets=1)

    assert isinstance(fig, Figure)


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


def test_forecast_run_seeds_each_forecast_step_by_date(monkeypatch):
    captured: list[tuple[int | None, int]] = []

    def forecast_from_fit(**kwargs):
        captured.append((kwargs["seed"], kwargs["n_rows"]))
        return object()

    _patch_runner(monkeypatch)
    monkeypatch.setattr("qraft.forecast.run.forecast_from_fit", forecast_from_fit)
    history = build_forecast_recipe_history(
        _market(),
        min_history=2,
        refit_every=12,
    )

    full_run = simulate_forecast_paths(
        _market(),
        history,
        min_history=2,
        forecast_cadence="every_bar",
        seed=7,
    )
    full_seeds = captured.copy()
    captured.clear()

    simulate_forecast_paths_from_snapshots(
        [forecast_snapshot_at(_market(), full_run.steps[2].as_of)],
        history,
        pipeline_config=history.pipeline_config,
        seed=7,
    )

    assert len({seed for seed, _ in full_seeds}) == len(full_seeds)
    assert captured == [full_seeds[2]]


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
    from qraft.backtest.inputs import precompute_inputs
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
        forecasts=Forecaster(refit_every=2),
        plan=InputPlan(expected_returns="forecast"),
    )

    assert selected_steps == [2, 4]
    assert len(table) == 4


def test_policy_input_single_final_bar_uses_prior_recipe(monkeypatch):
    from qraft.construction.inputs import forecast_run_for_source
    from qraft.forecast.forecaster import Forecaster

    selected_steps: list[int] = []
    applied: list[object] = []

    def create_forecast_recipe(**kwargs):
        recipe = object()
        selected_steps.append(kwargs["data"].height)
        return recipe

    def apply_forecast_recipe(recipe, *args, **kwargs):
        applied.append(recipe)
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
    snapshot = market.snapshot_at(market.trading_bars[-1], market.trading_bars[-1])

    run = forecast_run_for_source([snapshot], Forecaster(refit_every=1), market=market)

    assert selected_steps == [5]
    assert len(applied) == 1
    assert run.forecast_at(market.trading_bars[-1]) is not None


def test_recipe_history_only_builds_snapshots_for_refit_bars(monkeypatch):
    _patch_runner(monkeypatch)
    snapshot_dates: list[datetime] = []

    def forecast_snapshot_at(market, bar):
        from qraft.core.snapshot import forecast_snapshot_at as real_snapshot_at

        snapshot_dates.append(bar)
        return real_snapshot_at(market, bar)

    monkeypatch.setattr("qraft.forecast.run.forecast_snapshot_at", forecast_snapshot_at)

    build_forecast_recipe_history(
        _market(),
        min_history=2,
        refit_every=2,
    )

    assert snapshot_dates == [
        datetime(2024, 1, 2),
        datetime(2024, 1, 4),
    ]


def test_selection_reuses_precomputed_decision_snapshots(monkeypatch):
    import numpy as np

    from qraft.backtest.selection.grid_eval import run_selection_window
    from qraft.construction.optimization.inputs import InputPlan
    from qraft.construction.policies import EqualWeightPolicy

    class CountingMarket:
        def __init__(self, market):
            self._market = market
            self.snapshot_calls: list[datetime] = []

        def snapshot_at(self, t, t_next, *, step_size=1):
            self.snapshot_calls.append(t)
            return self._market.snapshot_at(t, t_next, step_size=step_size)

        def __getattr__(self, name):
            return getattr(self._market, name)

    class StaticProvider:
        def for_date(self, snapshot, step):
            from qraft.construction.optimization.inputs import OptimizerInputs

            assets = list(snapshot.universe.assets)
            n_assets = len(assets)
            return OptimizerInputs(
                assets=assets,
                mean=np.zeros((1, n_assets)),
                covariances=np.tile(np.eye(n_assets), (1, 1, 1)),
            )

    market = CountingMarket(_market())
    run_selection_window(
        market,
        EqualWeightPolicy(target_cash_weight=0.0, min_history=2),
        grid={},
        forecasts=StaticProvider(),
        schedule=RebalanceSchedule("every_bar"),
        plan=InputPlan(expected_returns="historical"),
    )

    assert market.snapshot_calls == [
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
        datetime(2024, 1, 4),
        datetime(2024, 1, 5),
    ]
