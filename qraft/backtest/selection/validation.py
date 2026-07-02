from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qraft.backtest.configs import (
    BacktestConfig,
    CombinatorialCVConfig,
    WalkForwardConfig,
)
from qraft.backtest.selection.candidates import apply_hyperparameters
from qraft.backtest.selection.combinatorial import (
    CombinatorialReport,
    combinatorial_from_evaluation,
)
from qraft.backtest.selection.evaluate import (
    SelectionInputSource,
    evaluate_candidate_grid,
)
from qraft.backtest.selection.evaluation import CandidateEvaluation
from qraft.backtest.selection.results import PolicyParams
from qraft.backtest.selection.select import Scorer
from qraft.backtest.selection.walkforward import (
    WalkForwardReport,
    walk_forward_from_evaluation,
)
from qraft.construction.optimization.inputs import InputPlan
from qraft.construction.policies import PolicyProtocol
from qraft.core.configs import SelectionMetric
from qraft.core.market import MarketData

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
    _evaluation_cache: dict[tuple[Any, ...], CandidateEvaluation] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def run(self) -> ValidationResult:
        if isinstance(self.cv_config, WalkForwardConfig):
            report = self.walk_forward(self.cv_config)
            return ValidationResult(
                report=report,
                base_policy=self.base_policy,
                selected_params=_walk_forward_selected_params(report),
            )
        if isinstance(self.cv_config, CombinatorialCVConfig):
            report = self.combinatorial(self.cv_config)
            return ValidationResult(
                report=report,
                base_policy=self.base_policy,
                selected_params=getattr(report, "selected_params", None),
            )
        raise ValueError("cv_config must be WalkForwardConfig or CombinatorialCVConfig")

    def walk_forward(self, cfg: WalkForwardConfig) -> WalkForwardReport:
        self.market.assert_backtest_safe()
        return walk_forward_from_evaluation(
            self._evaluation(metric=cfg.metric, risk_free_rate=cfg.risk_free_rate),
            walk_config=cfg,
            score=self.score,
        )

    def combinatorial(self, cfg: CombinatorialCVConfig) -> CombinatorialReport:
        self.market.assert_backtest_safe()
        return combinatorial_from_evaluation(
            self._evaluation(
                metric=cfg.cv_config.metric,
                risk_free_rate=cfg.cv_config.risk_free_rate,
            ),
            cv_config=cfg,
            score=self.score,
        )

    def _evaluation(
        self,
        *,
        metric: SelectionMetric,
        risk_free_rate: float,
    ) -> CandidateEvaluation:
        key = self._evaluation_cache_key(metric, risk_free_rate)
        if key not in self._evaluation_cache:
            self._evaluation_cache[key] = evaluate_candidate_grid(
                self.market,
                self.base_policy,
                self.grid,
                self.backtest_config,
                risk_free_rate,
                metric=metric,
                score=self.score,
                source=self.source,
                plan=self.plan,
            )
        return self._evaluation_cache[key]

    def _evaluation_cache_key(
        self,
        metric: SelectionMetric,
        risk_free_rate: float,
    ) -> tuple[Any, ...]:
        return (
            id(self.market),
            id(self.base_policy),
            _freeze_mapping(self.grid),
            id(self.source),
            self.plan,
            self.backtest_config,
            metric,
            risk_free_rate,
            id(self.score),
        )


def _walk_forward_selected_params(report: WalkForwardReport) -> PolicyParams | None:
    selected = [
        fold.selection.selected_params
        for fold in report.folds
        if fold.selection.selected_params is not None
    ]
    if not selected:
        return None
    return max(set(selected), key=selected.count)


def _freeze_mapping(
    mapping: Mapping[str, Sequence[Any]],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return tuple((key, tuple(values)) for key, values in sorted(mapping.items()))
