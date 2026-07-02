from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qraft.backtest.configs import (
    BacktestConfig,
    CombinatorialCVConfig,
    WalkForwardConfig,
)
from qraft.core.market import MarketData
from qraft.backtest.selection.combinatorial import (
    CombinatorialReport,
    combinatorial_purged,
)
from qraft.backtest.selection.evaluate import SelectionInputSource
from qraft.backtest.selection.select import Scorer
from qraft.backtest.selection.walkforward import WalkForwardReport, walk_forward
from qraft.construction.optimization.inputs import InputPlan
from qraft.construction.policies import PolicyProtocol


@dataclass(frozen=True, slots=True)
class Validation:
    market: MarketData
    base_policy: PolicyProtocol
    grid: Mapping[str, Sequence[Any]]
    source: SelectionInputSource | None = None
    plan: InputPlan | None = None
    cv_config: WalkForwardConfig | CombinatorialCVConfig = field(
        default_factory=WalkForwardConfig
    )
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    score: Scorer | None = None

    def run(self) -> WalkForwardReport | CombinatorialReport:
        self.market.assert_backtest_safe()
        if isinstance(self.cv_config, WalkForwardConfig):
            return walk_forward(
                self.market,
                self.base_policy,
                self.grid,
                source=self.source,
                plan=self.plan,
                walk_config=self.cv_config,
                backtest_config=self.backtest_config,
                score=self.score,
            )
        if isinstance(self.cv_config, CombinatorialCVConfig):
            return combinatorial_purged(
                self.market,
                self.base_policy,
                self.grid,
                source=self.source,
                plan=self.plan,
                cv_config=self.cv_config,
                backtest_config=self.backtest_config,
                score=self.score,
            )
        raise ValueError("cv_config must be WalkForwardConfig or CombinatorialCVConfig")
