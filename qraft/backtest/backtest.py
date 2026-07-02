from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.costs import CostModel
from qraft.backtest.execution import BacktestResult
from qraft.backtest.inputs import PolicyInputsProvider, PrecomputedInputsProvider
from qraft.core.market import MarketData
from qraft.backtest.simulator import precompute_inputs, run_backtest
from qraft.construction.optimization.inputs import InputPlan, PolicyInputs
from qraft.construction.policies import PolicyProtocol
from qraft.forecast.forecaster import ForecastSource

BacktestSource: TypeAlias = (
    ForecastSource | PolicyInputsProvider | dict[datetime, PolicyInputs]
)


@dataclass(frozen=True, slots=True)
class Backtest:
    market: MarketData
    policy: PolicyProtocol
    source: BacktestSource | None = None
    plan: InputPlan | None = None
    config: BacktestConfig = field(default_factory=BacktestConfig)
    costs: CostModel | None = None
    step_size: int = 1

    def run(self) -> BacktestResult:
        self.market.assert_backtest_safe()
        inputs = None
        if self.source is not None:
            table = precompute_inputs(
                self.market,
                self.config.schedule,
                self.policy.min_history,
                plan=self.plan,
                source=self.source,
                policy=self.policy,
                step_size=self.step_size,
            )
            inputs = PrecomputedInputsProvider(table)

        return run_backtest(
            self.market,
            self.policy,
            schedule=self.config.schedule,
            inputs=inputs,
            initial_cash=self.config.initial_cash,
            step_size=self.step_size,
            costs=self.costs,
        )
