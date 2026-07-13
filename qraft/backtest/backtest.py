from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias

from qraft.backtest.configs import BacktestConfig
from qraft.backtest.costs import CostModel
from qraft.backtest.engine.loop import run_backtest
from qraft.backtest.inputs import (
    OptimizerInputsProvider,
    PrecomputedInputsProvider,
    precompute_inputs,
)
from qraft.backtest.result import BacktestResult
from qraft.core.market import MarketData
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.policies import PolicyProtocol
from qraft.forecast.forecaster import ForecastSpec

BacktestSource: TypeAlias = (
    ForecastSpec | OptimizerInputsProvider | dict[datetime, OptimizerInputs]
)


@dataclass(frozen=True, slots=True)
class Backtest:
    market: MarketData
    policy: PolicyProtocol
    forecasts: BacktestSource | None = None
    config: BacktestConfig = field(default_factory=BacktestConfig)
    costs: CostModel | None = None
    step_size: int = 1

    def run(self) -> BacktestResult:
        inputs = None
        if self.forecasts is not None:
            table = precompute_inputs(
                self.market,
                self.config.schedule,
                self.policy.min_history,
                forecasts=self.forecasts,
                policy=self.policy,
                step_size=self.step_size,
            )
            if table or isinstance(self.forecasts, dict):
                inputs = PrecomputedInputsProvider(table)

        return run_backtest(
            self.market,
            self.policy,
            schedule=self.config.schedule,
            inputs=inputs,
            initial_cash=self.config.initial_cash,
            step_size=self.step_size,
            costs=self.costs,
            config=self.config,
        )
