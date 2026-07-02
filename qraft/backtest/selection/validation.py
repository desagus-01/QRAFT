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
from qraft.backtest.selection.results import CandidateResult, PolicyParams
from qraft.backtest.selection.scoring import Agg
from qraft.backtest.selection.select import Scorer, _conservatism, _eligible
from qraft.backtest.selection.splits import DateRange
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
            self._evaluation(risk_free_rate=cfg.risk_free_rate),
            walk_config=cfg,
            score=self.score,
        )

    def combinatorial(self, cfg: CombinatorialCVConfig) -> CombinatorialReport:
        self.market.assert_backtest_safe()
        return combinatorial_from_evaluation(
            self._evaluation(risk_free_rate=cfg.cv_config.risk_free_rate),
            cv_config=cfg,
            score=self.score,
        )

    def tune(
        self,
        report: WalkForwardReport | CombinatorialReport | None = None,
        *,
        metric: SelectionMetric | None = None,
        scorer: Scorer | None = None,
        agg: Agg = "mean",
    ) -> PolicyParams:
        if metric is None:
            metric = self._resolve_metric()
        if scorer is None:
            scorer = self.score
        if report is None:
            risk_free_rate = self._resolve_risk_free_rate()
            evaluation = self._evaluation(risk_free_rate=risk_free_rate)
            scores = evaluation.full_sample_scores(metric=metric, scorer=scorer)
            candidates = evaluation.candidate_results
        elif isinstance(report, WalkForwardReport):
            evaluation = report.evaluation or self._evaluation(
                risk_free_rate=self._resolve_risk_free_rate()
            )
            windows: dict[PolicyParams, list[DateRange]] = {}
            for fold_result in report.folds:
                params = fold_result.selection.selected_params
                if params is not None:
                    windows.setdefault(params, []).append(fold_result.fold.test)
            scores = evaluation.oos_scores(
                windows, metric=metric, scorer=scorer, agg=agg
            )
            candidates = evaluation.candidate_results
        elif isinstance(report, CombinatorialReport):
            evaluation = report.evaluation or self._evaluation(
                risk_free_rate=self._resolve_risk_free_rate()
            )
            windows = {}
            for fold, params in zip(report.folds, report.fold_selected_params):
                if params is not None:
                    windows.setdefault(params, []).extend(fold.test)
            scores = evaluation.oos_scores(
                windows, metric=metric, scorer=scorer, agg=agg
            )
            candidates = evaluation.candidate_results
        else:
            raise TypeError(
                f"Expected None, WalkForwardReport, or CombinatorialReport, "
                f"got {type(report).__name__}"
            )

        max_held_fraction = self._resolve_max_held_fraction()
        return _tune_select(scores, candidates, max_held_fraction)

    def tuned_policy(
        self,
        report: WalkForwardReport | CombinatorialReport | None = None,
        *,
        metric: SelectionMetric | None = None,
        scorer: Scorer | None = None,
        agg: Agg = "mean",
    ) -> PolicyProtocol:
        return apply_hyperparameters(
            self.base_policy,
            self.tune(report=report, metric=metric, scorer=scorer, agg=agg),
        )

    def _resolve_metric(self) -> SelectionMetric:
        if isinstance(self.cv_config, WalkForwardConfig):
            return self.cv_config.metric
        if isinstance(self.cv_config, CombinatorialCVConfig):
            return self.cv_config.cv_config.metric
        raise TypeError(f"Unexpected cv_config type: {type(self.cv_config).__name__}")

    def _resolve_risk_free_rate(self) -> float:
        if isinstance(self.cv_config, WalkForwardConfig):
            return self.cv_config.risk_free_rate
        if isinstance(self.cv_config, CombinatorialCVConfig):
            return self.cv_config.cv_config.risk_free_rate
        raise TypeError(f"Unexpected cv_config type: {type(self.cv_config).__name__}")

    def _resolve_max_held_fraction(self) -> float:
        if isinstance(self.cv_config, WalkForwardConfig):
            return self.cv_config.max_held_fraction
        if isinstance(self.cv_config, CombinatorialCVConfig):
            return self.cv_config.cv_config.max_held_fraction
        raise TypeError(f"Unexpected cv_config type: {type(self.cv_config).__name__}")

    def _evaluation(
        self,
        *,
        risk_free_rate: float,
    ) -> CandidateEvaluation:
        key = self._evaluation_cache_key(risk_free_rate)
        if key not in self._evaluation_cache:
            self._evaluation_cache[key] = evaluate_candidate_grid(
                self.market,
                self.base_policy,
                self.grid,
                self.backtest_config,
                risk_free_rate,
                source=self.source,
                plan=self.plan,
            )
        return self._evaluation_cache[key]

    def _evaluation_cache_key(
        self,
        risk_free_rate: float,
    ) -> tuple[Any, ...]:
        return (
            id(self.market),
            id(self.base_policy),
            _freeze_mapping(self.grid),
            id(self.source),
            self.plan,
            self.backtest_config,
            risk_free_rate,
        )


def _tune_select(
    scores: dict[PolicyParams, float],
    candidates: tuple[CandidateResult, ...],
    max_held_fraction: float,
) -> PolicyParams:
    candidate_map = {c.params: c for c in candidates}
    eligible: dict[PolicyParams, float] = {}
    for params, score in scores.items():
        c = candidate_map.get(params)
        if c is None or not _eligible(c, max_held_fraction):
            continue
        eligible[params] = score
    if not eligible:
        raise ValueError("No eligible candidate found for tuning.")
    return max(
        eligible,
        key=lambda p: (eligible[p], _conservatism(candidate_map[p])),
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
