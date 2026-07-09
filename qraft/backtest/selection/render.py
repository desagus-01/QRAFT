from __future__ import annotations

from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qraft.backtest.selection.reports import CombinatorialReport, WalkForwardReport


def plot_walk_forward_report(report: WalkForwardReport) -> Figure:
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    dates = list(report.oos_nav_dates)
    nav = np.asarray(report.oos_nav, dtype=float)

    ax = axes[0, 0]
    if nav.size:
        ax.plot(dates, nav, color="steelblue", linewidth=1.6)
        for fold_result in report.folds:
            ax.axvspan(
                fold_result.fold.test[0],
                fold_result.fold.test[1],
                color="steelblue",
                alpha=0.06,
            )
    ax.set_title("Stitched OOS NAV")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if nav.size:
        peak = np.maximum.accumulate(nav)
        drawdown = nav / peak - 1.0
        ax.fill_between(dates, 0.0, drawdown, color="crimson", alpha=0.35)
        ax.plot(dates, drawdown, color="crimson", linewidth=1.0)
    ax.set_title("Stitched OOS Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    fold_ids = np.arange(len(report.folds))
    test_returns = _column_or_nan(
        report.folds_df, "test_total_return", len(report.folds)
    )
    colors = ["forestgreen" if r >= 0 else "crimson" for r in test_returns]
    ax.bar(fold_ids, test_returns, color=colors, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("OOS Return By Fold")
    ax.set_xlabel("Fold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    train_sharpe = _column_or_nan(report.folds_df, "train_sharpe", len(report.folds))
    test_sharpe = _column_or_nan(report.folds_df, "test_sharpe", len(report.folds))
    ax.plot(fold_ids, train_sharpe, marker="o", label="Train selected")
    ax.plot(fold_ids, test_sharpe, marker="o", label="OOS")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Sharpe By Fold")
    ax.set_xlabel("Fold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    sharpe_decay = _column_or_nan(report.folds_df, "sharpe_decay", len(report.folds))
    colors = ["forestgreen" if d >= 0 else "crimson" for d in sharpe_decay]
    ax.bar(fold_ids, sharpe_decay, color=colors, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("OOS Sharpe Decay")
    ax.set_xlabel("Fold")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2, 1]
    if not report.selection_counts_df.is_empty():
        labels = report.selection_counts_df["selected_params"].to_list()
        values = report.selection_counts_df["n_folds"].to_list()
        y = np.arange(len(labels))
        ax.barh(y, values, color="slateblue", alpha=0.75)
        ax.set_yticks(y, [_truncate_label(label) for label in labels])
        ax.invert_yaxis()
    ax.set_title("Selected Parameter Stability")
    ax.set_xlabel("Folds selected")
    ax.grid(True, axis="x", alpha=0.3)

    _plot_diagnostics_bar(
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
        f"Deflated Sharpe: {_format_optional_float(report.deflated_sharpe)}",
        f"PBO: {_format_optional_pct(report.pbo)}",
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
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )

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
        nav = _path_nav(returns)
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

    _plot_diagnostics_bar(
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
        f"Median Sharpe: {_format_optional_float(report.median_sharpe)}",
        f"Sharpe IQR: {_format_optional_float(lo)} to {_format_optional_float(hi)}",
        f"Worst Sharpe: {_format_optional_float(report.worst_sharpe)}",
        f"Deflated Sharpe: {_format_optional_float(report.deflated_sharpe)}",
        f"PBO: {_format_optional_pct(report.pbo)}",
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
        f"median Sharpe {_format_optional_float(report.median_sharpe)}"
    )
    if report.deflated_sharpe is not None:
        title += f" | DSR {report.deflated_sharpe:.2f}"
    if report.pbo is not None:
        title += f" | PBO {report.pbo:.1%}"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def _plot_diagnostics_bar(
    ax: Axes,
    *,
    title: str,
    n_trials: int,
    deflated_sharpe: float | None,
    pbo: float | None,
) -> None:
    diagnostic_labels = ["Deflated\nSharpe", "PBO", "Trials"]
    diagnostic_values = [
        deflated_sharpe if deflated_sharpe is not None else np.nan,
        pbo if pbo is not None else np.nan,
        float(n_trials),
    ]
    colors = ["forestgreen", "crimson", "steelblue"]
    bars = ax.bar(diagnostic_labels, diagnostic_values, color=colors, alpha=0.75)
    for bar, value, label in zip(
        bars, diagnostic_values, diagnostic_labels, strict=True
    ):
        if np.isfinite(value):
            text = f"{value:.1%}" if label == "PBO" else f"{value:.2f}"
            if label == "Trials":
                text = f"{int(value)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def _column_or_nan(frame: pl.DataFrame, name: str, length: int) -> np.ndarray:
    if name not in frame.columns:
        return np.full(length, np.nan)
    return np.array(frame[name].to_list(), dtype=float)


def _truncate_label(label: str, max_len: int = 42) -> str:
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 3]}..."


def _path_nav(returns: Sequence[float]) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.array([], dtype=float)
    return np.concatenate(([1.0], np.cumprod(1.0 + returns)))


def _format_optional_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def _format_optional_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1%}"
