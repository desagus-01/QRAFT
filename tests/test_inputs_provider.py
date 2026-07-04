from datetime import datetime

import numpy as np
import polars as pl

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.inputs import (
    DateCache,
    PrecomputedInputsProvider,
)
from qraft.core.market import MarketData
from qraft.backtest.selection.evaluate import evaluate_candidate_grid
from qraft.backtest.simulator import (
    precompute_inputs,
)
from qraft.construction.inputs import build_policy_input_table
from qraft.construction.market_snapshot import MarketSnapshot
from qraft.construction.optimization.inputs import (
    InputPlan,
    PolicyInputs,
    RequiredPolicyInputs,
)
from qraft.core.panel import ScenarioPanel
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


class _Counting:
    def __init__(self) -> None:
        self.calls: list[datetime] = []

    def for_date(self, snapshot, step):
        self.calls.append(snapshot.t)
        return PolicyInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
        )


def test_datecache_builds_once_and_returns_identical_objects():
    inner = _Counting()
    cache = DateCache(inner)
    snaps = [_snap(datetime(2024, 1, d)) for d in (2, 3, 4)]
    first = [cache.for_date(s, i) for i, s in enumerate(snaps)]  # gamma=0 pass
    second = [cache.for_date(s, i) for i, s in enumerate(snaps)]  # gamma=1 pass
    assert inner.calls == [s.t for s in snaps]  # once per date
    assert all(a is b for a, b in zip(first, second))  # fair: same object


def test_precomputed_provider_serves_table():
    t = datetime(2024, 1, 2)
    inp = PolicyInputs.from_arrays(
        assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
    )
    prov = PrecomputedInputsProvider({t: inp})
    assert prov.for_date(_snap(t), 0) is inp


def test_astype_downcasts_only_heavy_arrays():
    inp = PolicyInputs.from_arrays(
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
            snapshot.t: PolicyInputs.from_arrays(
                assets=["A"], mean=np.ones((1, 1)), cash_return=snapshot.cash_rate
            )
            for snapshot in snapshots
        }

    monkeypatch.setattr(
        "qraft.backtest.simulator.build_policy_input_table",
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
        plan=InputPlan(),
        forecaster=object(),
    )
    inputs = table[datetime(2024, 1, 2)]

    np.testing.assert_allclose(inputs.cash_return, [0.001 / 100 / 360])
    assert captured["cash_return"] == 0.001 / 100 / 360


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
        return PolicyInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=kwargs["cash_return"]
        )

    monkeypatch.setattr(
        "qraft.construction.inputs.PolicyInputs.from_policy_sources",
        fake_from_policy_sources,
    )

    table = build_policy_input_table(
        [snapshot],
        [forecast],
        plan=InputPlan(risk="both"),
    )

    assert table.keys() == {snapshot.t}
    assert captured["forecasts"] is forecast
    assert captured["history"] is snapshot.history
    assert captured["cash_return"] == 0.002


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
            snapshot.t: PolicyInputs.from_arrays(
                assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
            )
            for snapshot in snapshots
        }

    monkeypatch.setattr(
        "qraft.backtest.simulator.build_policy_input_table",
        fake_build,
    )

    table = precompute_inputs(
        market,
        RebalanceSchedule("every_bar"),
        warmup=2,
        plan=InputPlan(),
        source=recipe_history,
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
        captured["recipe_history"] = kwargs["source"]
        return {
            dates[1]: PolicyInputs.from_arrays(
                assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
            )
        }

    def fake_evaluate(candidates, market_arg, inputs, **kwargs):
        captured["inputs"] = inputs
        return ()

    monkeypatch.setattr(
        "qraft.backtest.selection.evaluate.precompute_inputs",
        fake_precompute,
    )
    monkeypatch.setattr(
        "qraft.backtest.selection.evaluate.evaluate_candidates",
        fake_evaluate,
    )

    evaluation = evaluate_candidate_grid(
        market,
        policy,
        {},
        BacktestConfig(),
        risk_free_rate=0.0,
        source=recipe_history,
        plan=InputPlan(),
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
        return RequiredPolicyInputs(mean=True)
