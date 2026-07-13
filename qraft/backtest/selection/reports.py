from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from numpy.typing import NDArray

from qraft.backtest.result import PerformanceSummary
from qraft.backtest.selection.candidate_eval import CandidateEvaluation
from qraft.backtest.selection.results import PolicyParams, SelectionReport
from qraft.backtest.selection.splits import CombinatorialFold, Fold
from qraft.utils.helpers import (
    column_or_nan,
    format_optional_float,
    format_optional_pct,
    most_common,
    path_nav,
    plot_diagnostics_bar,
    safe_div,
    selection_concentration,
    truncate_label,
)
from qraft.utils.visuals import as_mpl_dates, format_date_axis


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

    def plot(self) -> Figure:
        return plot_walk_forward_report(self)

    @property
    def selected_params(self) -> PolicyParams | None:
        return _walk_forward_selected_params(self)

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
                        "vol_ratio": safe_div(
                            fold_result.oos_summary.annualised_vol,
                            selected.summary.annualised_vol,
                        ),
                    }
                )
            rows.append(row)
        return pl.DataFrame(rows)

    def summary_df(self) -> pl.DataFrame:
        row: dict[str, object] = {
            "n_folds": len(self.folds),
            "n_trials": self.n_trials,
            "deflated_sharpe": self.deflated_sharpe,
            "pbo": self.pbo,
        }
        if self.oos_summary is not None:
            row.update({f"oos_{k}": v for k, v in self.oos_summary.to_dict().items()})
        fold_rows = self.folds_df()
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
                row["selection_concentration"] = selection_concentration(selected)
                row["most_selected_params"] = most_common(selected)
        return pl.DataFrame([row])

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
        summary = self.summary_df()
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

    def plot(self) -> Figure:
        return plot_combinatorial_report(self)

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


def _mark_rebalances(ax, dates: list[datetime]) -> None:
    first = True
    for date in as_mpl_dates(dates):
        ax.axvline(
            date,
            color="red",
            alpha=0.18,
            linewidth=0.8,
            label="Rebalance" if first else None,
            zorder=0,
        )
        first = False


def _walk_forward_selected_params(report: WalkForwardReport) -> PolicyParams | None:
    selected = [
        fold.selection.selected_params
        for fold in report.folds
        if fold.selection.selected_params is not None
    ]
    if not selected:
        return None
    return max(set(selected), key=selected.count)


def plot_walk_forward_report(report: WalkForwardReport) -> Figure:
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    dates = as_mpl_dates(report.oos_nav_dates)
    nav = np.asarray(report.oos_nav, dtype=float)

    ax = axes[0, 0]
    if nav.size:
        ax.plot(dates, nav, color="steelblue", linewidth=1.6)
        for fold_result in report.folds:
            ax.axvspan(
                as_mpl_dates([fold_result.fold.test[0]])[0],
                as_mpl_dates([fold_result.fold.test[1]])[0],
                color="steelblue",
                alpha=0.06,
            )
        _mark_rebalances(ax, [fold_result.fold.test[0] for fold_result in report.folds])
        ax.legend(fontsize=8)
    ax.set_title("Stitched OOS NAV")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if nav.size:
        peak = np.maximum.accumulate(nav)
        drawdown = nav / peak - 1.0
        ax.fill_between(dates, 0.0, drawdown, color="crimson", alpha=0.35)
        ax.plot(dates, drawdown, color="crimson", linewidth=1.0)
        _mark_rebalances(ax, [fold_result.fold.test[0] for fold_result in report.folds])
    ax.set_title("Stitched OOS Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    fold_ids = np.arange(len(report.folds))
    test_returns = column_or_nan(
        report.folds_df(), "test_total_return", len(report.folds)
    )
    colors = ["forestgreen" if r >= 0 else "crimson" for r in test_returns]
    ax.bar(fold_ids, test_returns, color=colors, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("OOS Return By Fold")
    ax.set_xlabel("Fold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    train_sharpe = column_or_nan(report.folds_df(), "train_sharpe", len(report.folds))
    test_sharpe = column_or_nan(report.folds_df(), "test_sharpe", len(report.folds))
    ax.plot(fold_ids, train_sharpe, marker="o", label="Train selected")
    ax.plot(fold_ids, test_sharpe, marker="o", label="OOS")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Sharpe By Fold")
    ax.set_xlabel("Fold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    sharpe_decay = column_or_nan(report.folds_df(), "sharpe_decay", len(report.folds))
    colors = ["forestgreen" if d >= 0 else "crimson" for d in sharpe_decay]
    ax.bar(fold_ids, sharpe_decay, color=colors, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("OOS Sharpe Decay")
    ax.set_xlabel("Fold")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2, 1]
    selection_counts = report.selection_counts_df()
    if not selection_counts.is_empty():
        labels = selection_counts["selected_params"].to_list()
        values = selection_counts["n_folds"].to_list()
        y = np.arange(len(labels))
        ax.barh(y, values, color="slateblue", alpha=0.75)
        ax.set_yticks(y, [truncate_label(label) for label in labels])
        ax.invert_yaxis()
    ax.set_title("Selected Parameter Stability")
    ax.set_xlabel("Folds selected")
    ax.grid(True, axis="x", alpha=0.3)

    plot_diagnostics_bar(
        axes[3, 0],
        title="Multiple-Testing Diagnostics",
        n_trials=report.n_trials,
        deflated_sharpe=report.deflated_sharpe,
        pbo=report.pbo,
    )

    ax = axes[3, 1]
    ax.axis("off")
    diagnostic_text = [
        "Robustness read-through",
        f"Trials tested: {report.n_trials}",
        f"Deflated Sharpe: {format_optional_float(report.deflated_sharpe)}",
        f"PBO: {format_optional_pct(report.pbo)}",
    ]
    if report.deflated_sharpe is not None:
        diagnostic_text.append(
            "DSR adjusts the observed OOS Sharpe for multiple trials and non-normality."
        )
    if report.pbo is not None:
        diagnostic_text.append(
            "PBO estimates how often selection would pick a strategy that underperforms OOS."
        )
    ax.text(
        0.0,
        1.0,
        "\n".join(diagnostic_text),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        linespacing=1.5,
    )

    for ax in axes.flat[:2]:
        format_date_axis(ax)

    title = "Walk-Forward OOS Report"
    if report.oos_summary is not None:
        title += (
            f" | return {report.oos_summary.total_return:.1%}, "
            f"Sharpe {report.oos_summary.sharpe:.2f}, "
            f"max DD {report.oos_summary.max_drawdown:.1%}"
        )
    if report.deflated_sharpe is not None:
        title += f" | DSR {report.deflated_sharpe:.2f}"
    if report.pbo is not None:
        title += f" | PBO {report.pbo:.1%}"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_combinatorial_report(report: CombinatorialReport) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sharpes = np.asarray(report.path_sharpes, dtype=float)
    finite_sharpes = sharpes[np.isfinite(sharpes)]

    ax = axes[0, 0]
    if finite_sharpes.size:
        ax.hist(finite_sharpes, bins="auto", color="steelblue", alpha=0.75)
        ax.axvline(
            report.median_sharpe,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"Median {report.median_sharpe:.2f}",
        )
        ax.axvline(0.0, color="crimson", linewidth=0.9, alpha=0.8)
        ax.legend(fontsize=8)
    ax.set_title("CPCV Sharpe Distribution")
    ax.set_xlabel("Path Sharpe")
    ax.set_ylabel("Paths")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    for returns in report.path_returns:
        nav = path_nav(returns)
        if nav.size:
            ax.plot(nav, color="steelblue", linewidth=1.0, alpha=0.3)
    if report.paths:
        terminal = np.array([p.total_return + 1.0 for p in report.paths], dtype=float)
        if np.isfinite(terminal).any():
            ax.axhline(
                float(np.nanmedian(terminal)),
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Median terminal NAV",
            )
            ax.legend(fontsize=8)
    ax.set_title("CPCV Path Fan")
    ax.set_xlabel("Path period")
    ax.set_ylabel("Normalized NAV")
    ax.grid(True, alpha=0.3)

    plot_diagnostics_bar(
        axes[1, 0],
        title="PBO / DSR Diagnostics",
        n_trials=report.n_trials,
        deflated_sharpe=report.deflated_sharpe,
        pbo=report.pbo,
    )

    ax = axes[1, 1]
    ax.axis("off")
    lo, hi = report.sharpe_iqr
    diagnostic_text = [
        "CPCV robustness read-through",
        f"Paths: {report.n_paths}",
        f"Groups: {report.n_groups}",
        f"Test groups per fold: {report.n_test_groups}",
        f"Trials tested: {report.n_trials}",
        f"Median Sharpe: {format_optional_float(report.median_sharpe)}",
        f"Sharpe IQR: {format_optional_float(lo)} to {format_optional_float(hi)}",
        f"Worst Sharpe: {format_optional_float(report.worst_sharpe)}",
        f"Deflated Sharpe: {format_optional_float(report.deflated_sharpe)}",
        f"PBO: {format_optional_pct(report.pbo)}",
    ]
    if report.deflated_sharpe is not None:
        diagnostic_text.append(
            "DSR adjusts the observed path Sharpe for multiple trials and non-normality."
        )
    if report.pbo is not None:
        diagnostic_text.append(
            "PBO estimates how often selection would pick a strategy that underperforms OOS."
        )
    ax.text(
        0.0,
        1.0,
        "\n".join(diagnostic_text),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        linespacing=1.5,
    )

    title = (
        f"CPCV Report | paths {report.n_paths}, "
        f"median Sharpe {format_optional_float(report.median_sharpe)}"
    )
    if report.deflated_sharpe is not None:
        title += f" | DSR {report.deflated_sharpe:.2f}"
    if report.pbo is not None:
        title += f" | PBO {report.pbo:.1%}"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
