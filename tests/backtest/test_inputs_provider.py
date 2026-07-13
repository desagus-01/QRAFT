from datetime import datetime

import numpy as np
import polars as pl

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.engine.schedule import decision_points
from qraft.backtest.inputs import (
    PrecomputedInputsProvider,
    precompute_inputs,
)
from qraft.core.market import MarketData
from qraft.backtest.selection.grid_eval import evaluate_candidate_grid
from qraft.construction.inputs import build_optimizer_input_table
from qraft.construction.market_snapshot import MarketSnapshot
from qraft.construction.optimization.inputs import (
    InputPlan,
    OptimizerInputs,
    RequiredOptimizerInputs,
)
from qraft.core.panel import ScenarioPanel
from qraft.core.scenarios.transforms import Views
from qraft.core.scenarios.view_types import MeanView
from qraft.core.schedule import RebalanceSchedule
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths


def _snap(t: datetime, cash_rate: float = 0.0) -> MarketSnapshot:
    hist = ScenarioPanel.from_prices(
        pl.DataFrame({"date": [datetime(2024, 1, 1), t], "A": [9.0, 10.0]})
    )
    return MarketSnapshot(
        t=t,
        t_next=t,
        universe=AssetUniverse.factors_free(["A"]),
        history=hist,
        prices_t=np.array([10.0]),
        cash_rate=cash_rate,
    )


def test_precomputed_provider_serves_table():
    t = datetime(2024, 1, 2)
    inp = OptimizerInputs.from_arrays(
        assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
    )
    prov = PrecomputedInputsProvider({t: inp})
    assert prov.for_date(_snap(t), 0) is inp


def test_astype_downcasts_only_heavy_arrays():
    inp = OptimizerInputs.from_arrays(
        assets=["A", "B"],
        mean=np.ones((1, 2)),
        covariances=np.eye(2)[None],
        scenario_returns=np.ones((4, 1, 2)),
        scenario_probs=np.full(4, 0.25),
        cash_return=np.array([0.0]),
    )
    c = inp.astype(np.float32)
    assert c.covariances.dtype == np.float32
    assert c.cov_factor.dtype == np.float32
    assert c.scenario_returns.dtype == np.float32
    assert c.mean.dtype == np.float64  # small arrays preserved


def test_precompute_inputs_uses_snapshot_cash_rate(monkeypatch):
    captured = {}

    def fake_build(snapshots, forecast_source, **kwargs):
        snapshots = list(snapshots)
        target = next(snapshot for snapshot in snapshots if snapshot.t.day == 2)
        captured["cash_return"] = target.cash_rate
        return {
            snapshot.t: OptimizerInputs.from_arrays(
                assets=["A"], mean=np.ones((1, 1)), cash_return=snapshot.cash_rate
            )
            for snapshot in snapshots
        }

    monkeypatch.setattr(
        "qraft.backtest.inputs.build_optimizer_input_table",
        fake_build,
    )

    market = MarketData.from_prices(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, day) for day in range(1, 4)],
                "A": [18.0, 20.0, 22.0],
            }
        ),
        AssetUniverse.factors_free(["A"]),
        cash=pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "DFF": [0.0, 0.001],
            }
        ),
    )
    table = precompute_inputs(
        market,
        RebalanceSchedule("every_bar"),
        warmup=1,
        forecaster=object(),
    )
    inputs = table[datetime(2024, 1, 2)]

    np.testing.assert_allclose(inputs.cash_return, [0.001 / 100 / 360])
    assert captured["cash_return"] == 0.001 / 100 / 360


def test_precompute_inputs_accepts_materialized_points(monkeypatch):
    captured = {}

    def fake_build(snapshots, forecast_source, **kwargs):
        snapshots = list(snapshots)
        captured["snapshots"] = snapshots
        return {
            snapshot.t: OptimizerInputs.from_arrays(
                assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
            )
            for snapshot in snapshots
        }

    monkeypatch.setattr(
        "qraft.backtest.inputs.build_optimizer_input_table",
        fake_build,
    )

    dates = [datetime(2024, 1, day) for day in range(1, 4)]
    market = MarketData.from_prices(
        pl.DataFrame({"date": dates, "A": [18.0, 20.0, 22.0]}),
        AssetUniverse.factors_free(["A"]),
    )
    points = decision_points(market, RebalanceSchedule("every_bar"), warmup=1)

    table = precompute_inputs(market, points=points, forecaster=object())

    assert [snapshot.t for snapshot in captured["snapshots"]] == dates[:-1]
    assert table.keys() == set(dates[:-1])


def test_policy_input_table_accepts_supplied_forecasts(monkeypatch):
    snapshot = _snap(datetime(2024, 1, 2), cash_rate=0.002)
    forecast = ForecastPaths(
        asset_paths={"A": np.array([[10.5], [11.0]])},
        dates=pl.Series("date", [datetime(2024, 1, 3)]),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 10.0},
        universe=AssetUniverse.factors_free(["A"]),
    )
    captured = {}

    def fake_from_policy_sources(**kwargs):
        captured.update(kwargs)
        return OptimizerInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=kwargs["cash_return"]
        )

    monkeypatch.setattr(
        "qraft.construction.inputs.OptimizerInputs.from_policy_sources",
        fake_from_policy_sources,
    )

    table = build_optimizer_input_table(
        [snapshot],
        [forecast],
    )

    assert table.keys() == {snapshot.t}
    assert captured["forecasts"] is forecast
    assert captured["history"] is snapshot.history
    assert captured["cash_return"] == 0.002


def test_policy_input_table_uses_snapshot_applied_view_diagnostics(monkeypatch):
    market = MarketData.from_prices(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, day) for day in range(1, 4)],
                "A": [10.0, 11.0, 12.0],
            }
        ),
        AssetUniverse.factors_free(["A"]),
    ).with_views(
        (
            datetime(2024, 1, 3),
            datetime(2024, 1, 3),
            Views([MeanView("A", "==", 1.0 / 11.0)]),
        )
    )
    snapshot = market.snapshot_at(datetime(2024, 1, 3), datetime(2024, 1, 3))
    forecast = ForecastPaths(
        asset_paths={"A": np.array([[10.5], [11.0]])},
        dates=pl.Series("date", [datetime(2024, 1, 4)]),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 10.0},
        universe=AssetUniverse.factors_free(["A"]),
    )
    captured = {}

    class MarketWrapper:
        views = market.views

        def view_report(self, _t):
            raise AssertionError(
                "build_optimizer_input_table should use snapshot.applied_views"
            )

    market_wrapper = MarketWrapper()

    def fake_from_policy_sources(**kwargs):
        captured.update(kwargs)
        return OptimizerInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
        )

    monkeypatch.setattr(
        "qraft.construction.inputs.OptimizerInputs.from_policy_sources",
        fake_from_policy_sources,
    )

    build_optimizer_input_table([snapshot], [forecast], market=market_wrapper)

    assert snapshot.applied_views is not None
    assert captured["view_diagnostics"] is snapshot.applied_views.diagnostics


def test_policy_input_table_builds_historical_mean_when_policy_requires_mean():
    snapshot = _snap(datetime(2024, 1, 2), cash_rate=0.002)
    forecast = ForecastPaths(
        asset_paths={"A": np.array([[10.5], [11.0]])},
        dates=pl.Series("date", [datetime(2024, 1, 3)]),
        path_probs=np.array([0.5, 0.5]),
        initial_prices={"A": 10.0},
        universe=AssetUniverse.factors_free(["A"]),
    )

    class MeanPolicy:
        input_plan = InputPlan(expected_returns="historical")

        def required_inputs(self) -> RequiredOptimizerInputs:
            return RequiredOptimizerInputs(mean=True)

    table = build_optimizer_input_table(
        [snapshot],
        [forecast],
        policy=MeanPolicy(),
    )

    assert table.keys() == {snapshot.t}
    assert table[snapshot.t].mean is not None


def test_precompute_from_recipe_history_simulates_then_builds_inputs(monkeypatch):
    dates = [datetime(2024, 1, day) for day in range(1, 5)]
    market = MarketData.from_prices(
        pl.DataFrame({"date": dates, "A": [18.0, 20.0, 22.0, 24.0]}),
        AssetUniverse.factors_free(["A"]),
    )
    recipe_history = object()
    captured = {}

    def fake_build(snapshots, source_arg, **kwargs):
        snapshots = list(snapshots)
        captured["build"] = (snapshots, source_arg, kwargs)
        return {
            snapshot.t: OptimizerInputs.from_arrays(
                assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
            )
            for snapshot in snapshots
        }

    monkeypatch.setattr(
        "qraft.backtest.inputs.build_optimizer_input_table",
        fake_build,
    )

    table = precompute_inputs(
        market,
        RebalanceSchedule("every_bar"),
        warmup=2,
        forecasts=recipe_history,
    )

    assert captured["build"][1] is recipe_history
    assert [snapshot.t for snapshot in captured["build"][0]] == dates[1:-1]
    assert table.keys() == set(dates[1:-1])


def test_selection_grid_accepts_recipe_history_without_provider(monkeypatch):
    dates = [datetime(2024, 1, day) for day in range(1, 4)]
    market = MarketData.from_prices(
        pl.DataFrame({"date": dates, "A": [18.0, 20.0, 22.0]}),
        AssetUniverse.factors_free(["A"]),
    )
    policy = _DummyPolicy()
    recipe_history = object()
    captured = {}

    def fake_precompute(*args, **kwargs):
        captured["recipe_history"] = kwargs["forecasts"]
        return {
            dates[1]: OptimizerInputs.from_arrays(
                assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
            )
        }

    def fake_evaluate(candidates, market_arg, inputs, **kwargs):
        captured["inputs"] = inputs
        return ()

    monkeypatch.setattr(
        "qraft.backtest.selection.grid_eval.precompute_inputs",
        fake_precompute,
    )
    monkeypatch.setattr(
        "qraft.backtest.selection.grid_eval.evaluate_candidates",
        fake_evaluate,
    )

    evaluation = evaluate_candidate_grid(
        market,
        policy,
        {},
        BacktestConfig(),
        risk_free_rate=0.0,
        forecasts=recipe_history,
    )

    assert evaluation.candidate_results == ()
    assert evaluation.dates == [dates[1]]
    assert captured["recipe_history"] is recipe_history
    assert isinstance(captured["inputs"], PrecomputedInputsProvider)


class _DummyPolicy:
    min_history = 1

    def decide(self, state, inputs):
        raise NotImplementedError

    def required_inputs(self):
        return RequiredOptimizerInputs(mean=True)
