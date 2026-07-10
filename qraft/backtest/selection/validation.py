from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from qraft.backtest.configs import (
    BacktestConfig,
    CombinatorialCVConfig,
    WalkForwardConfig,
)
from qraft.backtest.selection.candidate_eval import CandidateEvaluation
from qraft.backtest.selection.candidates import apply_hyperparameters
from qraft.backtest.selection.grid_eval import (
    SelectionInputSource,
    evaluate_candidate_grid,
)
from qraft.backtest.selection.reports import (
    CombinatorialReport,
    ValidationReport,
    WalkForwardReport,
)
from qraft.backtest.selection.results import CandidateResult, PolicyParams
from qraft.backtest.selection.scoring import Agg, ScoreSpec
from qraft.backtest.selection.select import _conservatism, _eligible
from qraft.backtest.selection.splits import DateRange
from qraft.backtest.selection.validation_runs import (
    combinatorial_from_evaluation,
    walk_forward_from_evaluation,
)
from qraft.construction.optimization.inputs import InputPlan
from qraft.construction.policies import PolicyProtocol
from qraft.core.configs import SelectionMetric
from qraft.core.market import MarketData

TuneSelection = Literal["most_selected", "oos_score"]


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
    source: SelectionInputSource
    plan: InputPlan
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    score: ScoreSpec | None = None
    _evaluation_cache: dict[tuple[Any, ...], CandidateEvaluation] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def walk_forward(self, cfg: WalkForwardConfig | None = None) -> WalkForwardReport:
        if cfg is None:
            cfg = WalkForwardConfig()
        return walk_forward_from_evaluation(
            self._evaluation(risk_free_rate=cfg.risk_free_rate),
            walk_config=cfg,
            score=self.score,
        )

    def combinatorial(
        self, cfg: CombinatorialCVConfig | None = None
    ) -> CombinatorialReport:
        if cfg is None:
            cfg = CombinatorialCVConfig()
        return combinatorial_from_evaluation(
            self._evaluation(risk_free_rate=cfg.cv_config.risk_free_rate),
            cv_config=cfg,
            score=self.score,
        )

    def tune(
        self,
        report: ValidationReport,
        *,
        cfg: WalkForwardConfig | CombinatorialCVConfig | None = None,
        score: ScoreSpec | None = None,
        agg: Agg = "mean",
        selection: TuneSelection = "most_selected",
    ) -> ValidationResult:
        if cfg is None:
            cfg = _default_config_for_report(report)
        if score is None:
            score = self.score if self.score is not None else _config_metric(cfg)
        if isinstance(report, WalkForwardReport):
            if not isinstance(cfg, WalkForwardConfig):
                raise TypeError("cfg must be WalkForwardConfig for WalkForwardReport")
            if selection == "most_selected":
                return ValidationResult(
                    report=report,
                    base_policy=self.base_policy,
                    selected_params=report.selected_params,
                )
            evaluation = report.evaluation or self._evaluation(
                risk_free_rate=cfg.risk_free_rate
            )
            windows: dict[PolicyParams, list[DateRange]] = {}
            for fold_result in report.folds:
                params = fold_result.selection.selected_params
                if params is not None:
                    windows.setdefault(params, []).append(fold_result.fold.test)
            scores = evaluation.oos_scores(windows, score=score, agg=agg)
            candidates = evaluation.candidate_results
        elif isinstance(report, CombinatorialReport):
            if not isinstance(cfg, CombinatorialCVConfig):
                raise TypeError(
                    "cfg must be CombinatorialCVConfig for CombinatorialReport"
                )
            if selection == "most_selected":
                return ValidationResult(
                    report=report,
                    base_policy=self.base_policy,
                    selected_params=report.selected_params,
                )
            evaluation = report.evaluation or self._evaluation(
                risk_free_rate=cfg.cv_config.risk_free_rate
            )
            windows = {}
            for fold, params in zip(report.folds, report.fold_selected_params):
                if params is not None:
                    windows.setdefault(params, []).extend(fold.test)
            scores = evaluation.oos_scores(windows, score=score, agg=agg)
            candidates = evaluation.candidate_results
        else:
            raise TypeError(
                f"Expected WalkForwardReport or CombinatorialReport, "
                f"got {type(report).__name__}"
            )
        if selection != "oos_score":
            raise ValueError("selection must be 'most_selected' or 'oos_score'")

        max_held_fraction = _config_max_held_fraction(cfg)
        return ValidationResult(
            report=report,
            base_policy=self.base_policy,
            selected_params=_tune_select(scores, candidates, max_held_fraction),
        )

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
        if c is None or not _eligible(c, max_held_fraction) or not np.isfinite(score):
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


def _default_config_for_report(
    report: ValidationReport,
) -> WalkForwardConfig | CombinatorialCVConfig:
    if isinstance(report, WalkForwardReport):
        return WalkForwardConfig()
    if isinstance(report, CombinatorialReport):
        return CombinatorialCVConfig()
    raise TypeError(
        f"Expected WalkForwardReport or CombinatorialReport, got {type(report).__name__}"
    )


def _config_metric(cfg: WalkForwardConfig | CombinatorialCVConfig) -> SelectionMetric:
    if isinstance(cfg, WalkForwardConfig):
        return cfg.metric
    return cfg.cv_config.metric


def _config_max_held_fraction(cfg: WalkForwardConfig | CombinatorialCVConfig) -> float:
    if isinstance(cfg, WalkForwardConfig):
        return cfg.max_held_fraction
    return cfg.cv_config.max_held_fraction


def _freeze_mapping(
    mapping: Mapping[str, Sequence[Any]],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return tuple((key, tuple(values)) for key, values in sorted(mapping.items()))
