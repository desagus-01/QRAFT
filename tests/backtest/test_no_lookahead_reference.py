from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from qraft.backtest.engine.loop import run_backtest
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.policies import PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.forecast.forecast_paths import AssetUniverse


class HistoryMomentumPolicy:
    """Allocate to the asset selected from causal optimizer inputs."""

    name = "history_momentum"
    min_history = 2

    def decide(
        self,
        state: PortfolioState,
        optimizer_inputs: OptimizerInputs | None = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        assert optimizer_inputs is not None
        winner = int(np.argmax(optimizer_inputs.require_mean()[0]))
        weights = np.zeros(len(state.asset_order))
        weights[winner] = 1.0
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=weights,
            target_cash_weight=0.0,
        )


class HistoryMomentumInputs:
    def for_date(self, snapshot, step: int) -> OptimizerInputs:
        del step
        history = snapshot.history.values.select(snapshot.universe.assets).to_numpy()
        momentum = history[-1] - history[0]
        return OptimizerInputs.from_arrays(
            assets=snapshot.universe.assets,
            mean=momentum[None, :],
        )


def _market(a: list[float], b: list[float]) -> MarketData:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(a))]
    return MarketData.from_prices(
        pl.DataFrame({"date": dates, "A": a, "B": b}),
        AssetUniverse.factors_free(["A", "B"]),
    )


def test_mutating_future_prices_does_not_change_earlier_decisions() -> None:
    baseline = _market(
        [100, 102, 104, 106, 108, 110],
        [100, 101, 100, 99, 98, 97],
    )
    mutated = _market(
        [100, 102, 104, 1_000, 10, 2_000],
        [100, 101, 100, 1, 2_000, 1],
    )
    cutoff = baseline.trading_bars[2]

    baseline_result = run_backtest(
        baseline,
        HistoryMomentumPolicy(),
        schedule=RebalanceSchedule("every_bar"),
        inputs=HistoryMomentumInputs(),
        initial_cash=100.0,
    )
    mutated_result = run_backtest(
        mutated,
        HistoryMomentumPolicy(),
        schedule=RebalanceSchedule("every_bar"),
        inputs=HistoryMomentumInputs(),
        initial_cash=100.0,
    )
    baseline_periods = [p for p in baseline_result.periods if p.decision_bar <= cutoff]
    mutated_periods = [p for p in mutated_result.periods if p.decision_bar <= cutoff]

    assert [p.decision_bar for p in baseline_periods] == [
        p.decision_bar for p in mutated_periods
    ]
    for baseline_period, mutated_period in zip(
        baseline_periods, mutated_periods, strict=True
    ):
        np.testing.assert_array_equal(
            baseline_period.decision.target_weights_risk,
            mutated_period.decision.target_weights_risk,
        )
        np.testing.assert_allclose(
            baseline_period.executed_share_trades,
            mutated_period.executed_share_trades,
        )
