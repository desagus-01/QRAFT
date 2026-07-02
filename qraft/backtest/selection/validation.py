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
from qraft.backtest.selection.candidates import apply_hyperparameters
from qraft.backtest.selection.evaluate import SelectionInputSource
from qraft.backtest.selection.results import PolicyParams
from qraft.backtest.selection.select import Scorer
from qraft.backtest.selection.walkforward import WalkForwardReport, walk_forward
from qraft.construction.optimization.inputs import InputPlan
from qraft.construction.policies import PolicyProtocol

ValidationReport = WalkForwardReport | CombinatorialReport


@dataclass(frozen=True, slots=True)
class ValidationResult:
    report: ValidationReport
    base_policy: PolicyProtocol
    selected_params: PolicyParams | None

    @property
    def selected_policy(self) -> PolicyProtocol:
        if self.selected_params is None:
            raise ValueError("Validation did not select a policy candidate.")
        return apply_hyperparameters(self.base_policy, self.selected_params)

    def __getattr__(self, name: str):
        return getattr(self.report, name)


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

    def run(self) -> ValidationResult:
        self.market.assert_backtest_safe()
        if isinstance(self.cv_config, WalkForwardConfig):
            report = walk_forward(
                self.market,
                self.base_policy,
                self.grid,
                source=self.source,
                plan=self.plan,
                walk_config=self.cv_config,
                backtest_config=self.backtest_config,
                score=self.score,
            )
            return ValidationResult(
                report=report,
                base_policy=self.base_policy,
                selected_params=_walk_forward_selected_params(report),
            )
        if isinstance(self.cv_config, CombinatorialCVConfig):
            report = combinatorial_purged(
                self.market,
                self.base_policy,
                self.grid,
                source=self.source,
                plan=self.plan,
                cv_config=self.cv_config,
                backtest_config=self.backtest_config,
                score=self.score,
            )
            return ValidationResult(
                report=report,
                base_policy=self.base_policy,
                selected_params=getattr(report, "selected_params", None),
            )
        raise ValueError("cv_config must be WalkForwardConfig or CombinatorialCVConfig")


def _walk_forward_selected_params(report: WalkForwardReport) -> PolicyParams | None:
    selected = [
        fold.selection.selected_params
        for fold in report.folds
        if fold.selection.selected_params is not None
    ]
    if not selected:
        return None
    return max(set(selected), key=selected.count)
