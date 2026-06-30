from datetime import datetime

import numpy as np
import polars as pl

from qraft.backtest.inputs import (
    DateCache,
    ForecastInputsProvider,
    PrecomputedInputsProvider,
)
from qraft.construction.market_snapshot import MarketSnapshot
from qraft.construction.inputs import build_policy_input_table
from qraft.construction.optimization.moments import PolicyInputConfig
from qraft.construction.optimization.moments import PolicyInputs
from qraft.core.panel import ScenarioPanel
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


def test_forecast_provider_uses_snapshot_cash_rate(monkeypatch):
    provider = ForecastInputsProvider(PolicyInputConfig(cash_path="unused.csv"))
    captured = {}

    def fake_build(snapshots, **kwargs):
        snapshots = list(snapshots)
        captured["cash_return"] = snapshots[0].cash_rate
        return {
            snapshots[0].t: PolicyInputs.from_arrays(
                assets=["A"], mean=np.ones((1, 1)), cash_return=snapshots[0].cash_rate
            )
        }

    monkeypatch.setattr(
        "qraft.backtest.inputs.forecast_policy_input_table",
        fake_build,
    )

    inputs = provider.for_date(_snap(datetime(2024, 1, 2), cash_rate=0.001), 1)

    np.testing.assert_allclose(inputs.cash_return, [0.001])
    assert captured["cash_return"] == 0.001


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
        input_config=PolicyInputConfig(cash_path="unused.csv"),
        risk_source="both",
    )

    assert table.keys() == {snapshot.t}
    assert captured["forecasts"] is forecast
    assert captured["history"] is snapshot.history
    assert captured["cash_return"] == 0.002
