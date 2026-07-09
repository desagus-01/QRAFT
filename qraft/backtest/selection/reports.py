from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.backtest.result import PerformanceSummary
from qraft.backtest.selection.candidate_eval import CandidateEvaluation
from qraft.backtest.selection.results import PolicyParams, SelectionReport
from qraft.backtest.selection.splits import CombinatorialFold, Fold


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold: Fold
    selection: SelectionReport
    oos_summary: PerformanceSummary | None


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    folds: tuple[FoldResult, ...]
    oos_summary: PerformanceSummary | None
    oos_nav_dates: list[datetime]
    oos_nav: NDArray[np.floating]
    evaluation: CandidateEvaluation | None = None
    n_trials: int = 0
    deflated_sharpe: float | None = None
    pbo: float | None = None

    def require(self, *, max_pbo: float | None = None) -> "WalkForwardReport":
        if max_pbo is not None and self.pbo is not None and self.pbo > max_pbo:
            raise ValueError(
                f"Walk-forward PBO {self.pbo:.3f} exceeds required maximum "
                f"{max_pbo:.3f}."
            )
        return self

    @property
    def selected_params(self):
        return _walk_forward_selected_params(self)

    @property
    def folds_df(self) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for i, fold_result in enumerate(self.folds):
            selected = fold_result.selection.selected
            row: dict[str, object] = {
                "fold": i,
                "train_start": fold_result.fold.train[0],
                "train_end": fold_result.fold.train[1],
                "test_start": fold_result.fold.test[0],
                "test_end": fold_result.fold.test[1],
                "selected_params": str(fold_result.selection.selected_params or ""),
                "selection_rule": fold_result.selection.rule,
                "n_candidates": len(fold_result.selection.candidates),
                "n_successful_candidates": sum(
                    c.summary is not None for c in fold_result.selection.candidates
                ),
            }
            if selected is not None and selected.summary is not None:
                row.update(
                    {
                        f"train_{key}": value
                        for key, value in selected.summary.to_dict().items()
                    }
                )
            if fold_result.oos_summary is not None:
                row.update(
                    {
                        f"test_{key}": value
                        for key, value in fold_result.oos_summary.to_dict().items()
                    }
                )
            if selected is not None and selected.summary and fold_result.oos_summary:
                row.update(
                    {
                        "sharpe_decay": fold_result.oos_summary.sharpe
                        - selected.summary.sharpe,
                        "return_decay": fold_result.oos_summary.annualised_return
                        - selected.summary.annualised_return,
                        "vol_ratio": _safe_div(
                            fold_result.oos_summary.annualised_vol,
                            selected.summary.annualised_vol,
                        ),
                    }
                )
            rows.append(row)
        return pl.DataFrame(rows)

    @property
    def summary_df(self) -> pl.DataFrame:
        row: dict[str, object] = {
            "n_folds": len(self.folds),
            "n_trials": self.n_trials,
            "deflated_sharpe": self.deflated_sharpe,
            "pbo": self.pbo,
        }
        if self.oos_summary is not None:
            row.update({f"oos_{k}": v for k, v in self.oos_summary.to_dict().items()})
        fold_rows = self.folds_df
        if not fold_rows.is_empty():
            for col in (
                "train_sharpe",
                "test_sharpe",
                "sharpe_decay",
                "test_total_return",
                "test_max_drawdown",
                "test_hit_rate",
                "test_held_fraction",
            ):
                if col in fold_rows.columns:
                    values = fold_rows[col].drop_nulls()
                    if not values.is_empty():
                        row[f"avg_{col}"] = values.mean()
                        row[f"median_{col}"] = values.median()
            if "selected_params" in fold_rows.columns:
                selected = [p for p in fold_rows["selected_params"].to_list() if p]
                row["n_unique_selected"] = len(set(selected))
                row["selection_concentration"] = _selection_concentration(selected)
                row["most_selected_params"] = _most_common(selected)
        return pl.DataFrame([row])

    @property
    def selected_params_df(self) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for i, fold_result in enumerate(self.folds):
            params = fold_result.selection.selected_params
            row: dict[str, object] = {
                "fold": i,
                "test_start": fold_result.fold.test[0],
                "test_end": fold_result.fold.test[1],
                "selected_params": str(params or ""),
            }
            if params is not None:
                row.update(params.as_dict())
            if fold_result.oos_summary is not None:
                row.update(
                    {
                        "test_total_return": fold_result.oos_summary.total_return,
                        "test_sharpe": fold_result.oos_summary.sharpe,
                        "test_max_drawdown": fold_result.oos_summary.max_drawdown,
                    }
                )
            rows.append(row)
        return pl.DataFrame(rows)

    @property
    def diagnostics_df(self) -> pl.DataFrame:
        rows = [
            ("folds", len(self.folds)),
            ("trials", self.n_trials),
            ("deflated_sharpe", self.deflated_sharpe),
            ("pbo", self.pbo),
        ]
        if self.oos_summary is not None:
            rows.extend(
                [
                    ("oos_total_return", self.oos_summary.total_return),
                    ("oos_sharpe", self.oos_summary.sharpe),
                    ("oos_max_drawdown", self.oos_summary.max_drawdown),
                    ("oos_hit_rate", self.oos_summary.hit_rate),
                ]
            )
        summary = self.summary_df
        if not summary.is_empty():
            row = summary.row(0, named=True)
            for key in (
                "avg_sharpe_decay",
                "median_sharpe_decay",
                "selection_concentration",
                "n_unique_selected",
                "most_selected_params",
            ):
                if key in row:
                    rows.append((key, row[key]))
        return pl.DataFrame(rows, schema=["metric", "value"], orient="row")

    @property
    def selection_counts_df(self) -> pl.DataFrame:
        selected = [
            str(fr.selection.selected_params or "")
            for fr in self.folds
            if fr.selection.selected_params is not None
        ]
        counts = {params: selected.count(params) for params in set(selected)}
        rows = [
            {
                "selected_params": params,
                "n_folds": count,
                "frequency": count / len(selected),
            }
            for params, count in sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            )
        ]
        return pl.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class CombinatorialReport:
    paths: tuple[PerformanceSummary, ...]
    path_sharpes: NDArray[np.floating]
    n_groups: int
    n_test_groups: int
    purge: int
    embargo: int
    n_trials: int
    evaluation: CandidateEvaluation | None = None
    folds: tuple[CombinatorialFold, ...] = ()
    path_returns: tuple[NDArray[np.floating], ...] = ()
    selected_params: PolicyParams | None = None
    fold_selected_params: tuple[PolicyParams | None, ...] = ()
    pbo: float | None = None
    deflated_sharpe: float | None = None

    def require(self, *, max_pbo: float | None = None) -> "CombinatorialReport":
        if max_pbo is not None and self.pbo is not None and self.pbo > max_pbo:
            raise ValueError(
                f"Combinatorial PBO {self.pbo:.3f} exceeds required maximum "
                f"{max_pbo:.3f}."
            )
        return self

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    @property
    def median_sharpe(self) -> float:
        return (
            float(np.median(self.path_sharpes)) if self.path_sharpes.size else math.nan
        )

    @property
    def sharpe_iqr(self) -> tuple[float, float]:
        if self.path_sharpes.size == 0:
            return math.nan, math.nan
        return (
            float(np.quantile(self.path_sharpes, 0.25)),
            float(np.quantile(self.path_sharpes, 0.75)),
        )

    @property
    def worst_sharpe(self) -> float:
        return float(self.path_sharpes.min()) if self.path_sharpes.size else math.nan

    def summary_df(self) -> pl.DataFrame:
        lo, hi = self.sharpe_iqr
        return pl.DataFrame(
            [
                {
                    "n_paths": self.n_paths,
                    "n_groups": self.n_groups,
                    "n_test_groups": self.n_test_groups,
                    "n_trials": self.n_trials,
                    "median_sharpe": self.median_sharpe,
                    "sharpe_q25": lo,
                    "sharpe_q75": hi,
                    "worst_sharpe": self.worst_sharpe,
                    "deflated_sharpe": self.deflated_sharpe,
                    "pbo": self.pbo,
                }
            ]
        )


ValidationReport = WalkForwardReport | CombinatorialReport


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _selection_concentration(selected: Sequence[str]) -> float:
    if not selected:
        return 0.0
    counts = [selected.count(params) for params in set(selected)]
    return max(counts) / len(selected)


def _most_common(selected: Sequence[str]) -> str:
    if not selected:
        return ""
    return max(set(selected), key=selected.count)


def _walk_forward_selected_params(report: WalkForwardReport):
    selected = [
        fold.selection.selected_params
        for fold in report.folds
        if fold.selection.selected_params is not None
    ]
    if not selected:
        return None
    return max(set(selected), key=selected.count)
