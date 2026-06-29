import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.backtest.execution import BacktestResult
from qraft.backtest.inputs import PolicyInputsProvider, PrecomputedInputsProvider
from qraft.backtest.market import MarketData
from qraft.backtest.metrics import PerformanceSummary
from qraft.backtest.selection.candidates import expand_candidates
from qraft.backtest.selection.evaluate import _shared_warmup, evaluate_candidates
from qraft.backtest.selection.results import (
    CandidateResult,
    PolicyParams,
    SelectionReport,
)
from qraft.backtest.selection.select import Scorer, select_candidate
from qraft.backtest.selection.significance import (
    deflated_sharpe as compute_deflated_sharpe,
)
from qraft.backtest.selection.significance import (
    prob_of_backtest_overfit as compute_pbo,
)
from qraft.backtest.simulator import precompute_inputs
from qraft.construction.policies import PolicyProtocol
from qraft.core import metrics
from qraft.core.configs import BacktestConfig, WalkForwardConfig


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward split, as inclusive ``(start, end)`` rebalance dates."""

    train: tuple[datetime, datetime]
    test: tuple[datetime, datetime]


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
    n_trials: int = 0
    deflated_sharpe: float | None = None
    pbo: float | None = None

    @property
    def folds_df(self) -> pl.DataFrame:
        """One row per fold with selected params and train/test metrics."""
        rows: list[dict[str, Any]] = []
        for i, fold_result in enumerate(self.folds):
            selected = fold_result.selection.selected
            row: dict[str, Any] = {
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
        """Single-row walk-forward summary with OOS and selection diagnostics."""
        row: dict[str, Any] = {
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
                unique_selected = set(selected)
                row["n_unique_selected"] = len(unique_selected)
                row["selection_concentration"] = _selection_concentration(selected)
                row["most_selected_params"] = _most_common(selected)
        return pl.DataFrame([row])

    @property
    def selected_params_df(self) -> pl.DataFrame:
        """Fold-by-fold selected parameter values and OOS validation metrics."""
        rows: list[dict[str, Any]] = []
        for i, fold_result in enumerate(self.folds):
            params = fold_result.selection.selected_params
            row: dict[str, Any] = {
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
        """Compact diagnostics useful for judging robustness, not just return."""
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
        """Frequency table for selected parameter sets across folds."""
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

    def plot(self):
        """Plot a walk-forward research dashboard. Requires matplotlib."""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        fig, axes = plt.subplots(4, 2, figsize=(14, 13))
        dates = self.oos_nav_dates
        nav = np.asarray(self.oos_nav, dtype=float)
        folds_df = self.folds_df

        ax = axes[0, 0]
        if nav.size:
            ax.plot(dates, nav, color="steelblue", linewidth=1.6)
            for fold_result in self.folds:
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
        fold_ids = np.arange(len(self.folds))
        test_returns = _column_or_nan(folds_df, "test_total_return", len(self.folds))
        colors = ["forestgreen" if r >= 0 else "crimson" for r in test_returns]
        ax.bar(fold_ids, test_returns, color=colors, alpha=0.75)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("OOS Return By Fold")
        ax.set_xlabel("Fold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1, 1]
        train_sharpe = _column_or_nan(folds_df, "train_sharpe", len(self.folds))
        test_sharpe = _column_or_nan(folds_df, "test_sharpe", len(self.folds))
        ax.plot(fold_ids, train_sharpe, marker="o", label="Train selected")
        ax.plot(fold_ids, test_sharpe, marker="o", label="OOS")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Sharpe By Fold")
        ax.set_xlabel("Fold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        sharpe_decay = _column_or_nan(folds_df, "sharpe_decay", len(self.folds))
        colors = ["forestgreen" if d >= 0 else "crimson" for d in sharpe_decay]
        ax.bar(fold_ids, sharpe_decay, color=colors, alpha=0.75)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("OOS Sharpe Decay")
        ax.set_xlabel("Fold")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[2, 1]
        counts = self.selection_counts_df
        if not counts.is_empty():
            labels = counts["selected_params"].to_list()
            values = counts["n_folds"].to_list()
            y = np.arange(len(labels))
            ax.barh(y, values, color="slateblue", alpha=0.75)
            ax.set_yticks(y, [_truncate_label(label) for label in labels])
            ax.invert_yaxis()
        ax.set_title("Selected Parameter Stability")
        ax.set_xlabel("Folds selected")
        ax.grid(True, axis="x", alpha=0.3)

        ax = axes[3, 0]
        diagnostic_labels = ["Deflated\nSharpe", "PBO", "Trials"]
        diagnostic_values = [
            self.deflated_sharpe if self.deflated_sharpe is not None else np.nan,
            self.pbo if self.pbo is not None else np.nan,
            float(self.n_trials),
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
        ax.set_title("Multiple-Testing Diagnostics")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[3, 1]
        ax.axis("off")
        diagnostic_text = [
            "Robustness read-through",
            f"Trials tested: {self.n_trials}",
            f"Deflated Sharpe: {_format_optional_float(self.deflated_sharpe)}",
            f"PBO: {_format_optional_pct(self.pbo)}",
        ]
        if self.deflated_sharpe is not None:
            diagnostic_text.append(
                "DSR adjusts the observed OOS Sharpe for multiple trials and non-normality."
            )
        if self.pbo is not None:
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
        if self.oos_summary is not None:
            title += (
                f" | return {self.oos_summary.total_return:.1%}, "
                f"Sharpe {self.oos_summary.sharpe:.2f}, "
                f"max DD {self.oos_summary.max_drawdown:.1%}"
            )
        if self.deflated_sharpe is not None:
            title += f" | DSR {self.deflated_sharpe:.2f}"
        if self.pbo is not None:
            title += f" | PBO {self.pbo:.1%}"
        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()
        return fig


def walk_forward_folds(
    dates: Sequence[datetime],
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    anchored: bool = False,
) -> list[Fold]:
    """Walk-forward folds over an ordered list of rebalance ``dates``."""
    if train_size < 1 or test_size < 1 or embargo < 0:
        raise ValueError("train_size, test_size must be >= 1 and embargo >= 0")
    advance = test_size if step is None else step
    if advance < 1:
        raise ValueError("step must be >= 1")

    ordered = list(dates)
    n = len(ordered)
    folds: list[Fold] = []
    start = 0
    while True:
        train_hi = start + train_size - 1
        test_lo = train_hi + 1 + embargo
        test_hi = test_lo + test_size - 1
        if test_hi >= n:
            break
        train_lo = 0 if anchored else start
        folds.append(
            Fold(
                train=(ordered[train_lo], ordered[train_hi]),
                test=(ordered[test_lo], ordered[test_hi]),
            )
        )
        start += advance
    return folds


def walk_forward(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[Any]],
    provider: PolicyInputsProvider,
    *,
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig = BacktestConfig(),
    score: Scorer | None = None,
) -> WalkForwardReport:
    """Walk-forward gamma-selection: run candidates once, then per fold select on
    the train window and score the choice on the held-out test window.

    Candidates are backtested ONCE over the full calendar; folds reuse those
    causal results by slicing (no re-running). The stitched OOS curve is the
    realised, look-ahead-free track record of the rolling selection.
    """
    full, dates = _evaluate_full_candidates(
        market,
        base_policy,
        grid,
        provider,
        backtest_config,
        walk_config.risk_free_rate,
    )

    folds = walk_forward_folds(
        dates,
        train_size=walk_config.train_size,
        test_size=walk_config.test_size,
        step=walk_config.fold_step,
        embargo=walk_config.embargo,
        anchored=walk_config.anchored,
    )
    fold_results, oos_returns, oos_dates = _run_folds(
        full, folds, walk_config, backtest_config, score
    )

    nav, nav_dates, summary = _stitch(
        oos_returns,
        oos_dates,
        folds,
        backtest_config.initial_cash,
        backtest_config.periods_per_year,
        walk_config.risk_free_rate,
    )
    n_trials, dsr, pbo_value = _compute_walk_forward_diagnostics(
        full, oos_returns, walk_config, backtest_config
    )

    return WalkForwardReport(
        folds=tuple(fold_results),
        oos_summary=summary,
        oos_nav_dates=nav_dates,
        oos_nav=nav,
        n_trials=n_trials,
        deflated_sharpe=dsr,
        pbo=pbo_value,
    )


def _evaluate_full_candidates(
    market: MarketData,
    base_policy: PolicyProtocol,
    grid: Mapping[str, Sequence[Any]],
    provider: PolicyInputsProvider,
    backtest_config: BacktestConfig,
    risk_free_rate: float,
) -> tuple[Sequence[CandidateResult], list[datetime]]:
    candidates = expand_candidates(base_policy, grid)
    warmup = _shared_warmup(candidates)
    table = precompute_inputs(
        market,
        backtest_config.schedule,
        provider,
        warmup,
    )
    shared = PrecomputedInputsProvider(table)
    full = evaluate_candidates(
        candidates,
        market,
        shared,
        schedule=backtest_config.schedule,
        initial_cash=backtest_config.initial_cash,
        periods_per_year=backtest_config.periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    return full, sorted(table)


def _select_fold_candidate(
    full: Sequence[CandidateResult],
    fold: Fold,
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
    score: Scorer | None,
) -> SelectionReport:
    train_scores = [
        _windowed_result(
            cr,
            fold.train,
            backtest_config.periods_per_year,
            walk_config.risk_free_rate,
        )
        for cr in full
    ]
    return select_candidate(
        train_scores,
        metric=walk_config.metric,
        max_held_fraction=walk_config.max_held_fraction,
        score=score,
    )


def _score_oos_fold(
    full: Sequence[CandidateResult],
    fold: Fold,
    selection: SelectionReport,
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
) -> tuple[PerformanceSummary | None, NDArray[np.floating] | None, list[datetime]]:
    if selection.selected_params is None:
        return None, None, []
    winner = _find(full, selection.selected_params)
    assert winner.backtest is not None
    test_result = winner.backtest.window(fold.test[0], fold.test[1])
    oos_summary = PerformanceSummary.from_backtest(
        test_result,
        active_only=False,
        periods_per_year=backtest_config.periods_per_year,
        risk_free_rate=walk_config.risk_free_rate,
    )
    rets = metrics.returns_from_nav(test_result.nav)
    if not rets.size:
        return oos_summary, None, []
    return oos_summary, rets, list(test_result.nav_dates[1:])


def _run_folds(
    full: Sequence[CandidateResult],
    folds: Sequence[Fold],
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
    score: Scorer | None,
) -> tuple[list[FoldResult], list[NDArray[np.floating]], list[datetime]]:
    fold_results: list[FoldResult] = []
    oos_returns: list[NDArray[np.floating]] = []
    oos_dates: list[datetime] = []
    for fold in folds:
        selection = _select_fold_candidate(
            full, fold, walk_config, backtest_config, score
        )
        oos_summary, rets, dates = _score_oos_fold(
            full, fold, selection, walk_config, backtest_config
        )
        if rets is not None:
            oos_returns.append(rets)
            oos_dates.extend(dates)
        fold_results.append(FoldResult(fold, selection, oos_summary))
    return fold_results, oos_returns, oos_dates


def _compute_walk_forward_diagnostics(
    full: Sequence[CandidateResult],
    oos_returns: Sequence[NDArray[np.floating]],
    walk_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
) -> tuple[int, float | None, float | None]:
    trial_sharpes = [
        cr.summary.sharpe / math.sqrt(backtest_config.periods_per_year)
        for cr in full
        if cr.summary is not None
    ]
    n_trials = len(trial_sharpes)
    dsr: float | None = None
    pbo_value: float | None = None
    if oos_returns and n_trials >= 2:
        dsr = compute_deflated_sharpe(np.concatenate(oos_returns), trial_sharpes)
        perf = _block_returns(full, walk_config.pbo_blocks)
        if perf is not None:
            pbo_value = compute_pbo(perf)
    return n_trials, dsr, pbo_value


def _windowed_result(
    result: CandidateResult,
    window: tuple[datetime, datetime],
    periods_per_year: float,
    risk_free_rate: float,
) -> CandidateResult:
    """Re-score a full-run candidate over a sub-window (a leakage-free slice)."""
    if result.failure is not None or result.backtest is None:
        return CandidateResult(params=result.params, failure=result.failure)
    windowed = result.backtest.window(window[0], window[1])
    summary = PerformanceSummary.from_backtest(
        windowed,
        active_only=False,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    return CandidateResult(params=result.params, summary=summary, backtest=windowed)


def _find(results: Sequence[CandidateResult], params: PolicyParams) -> CandidateResult:
    return next(r for r in results if r.params == params)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _column_or_nan(df: pl.DataFrame, column: str, n: int) -> list[float]:
    if df.is_empty() or column not in df.columns:
        return [float("nan")] * n
    return [float(v) if v is not None else float("nan") for v in df[column].to_list()]


def _selection_concentration(selected: Sequence[str]) -> float:
    if not selected:
        return 0.0
    counts = [selected.count(params) for params in set(selected)]
    return max(counts) / len(selected)


def _most_common(selected: Sequence[str]) -> str:
    if not selected:
        return ""
    return max(set(selected), key=selected.count)


def _truncate_label(label: str, max_len: int = 36) -> str:
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 3]}..."


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_optional_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _block_returns(
    full: Sequence[CandidateResult], n_blocks: int
) -> NDArray[np.floating] | None:
    """Build the ``(n_blocks, n_candidates)`` mean-return matrix PBO needs."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    runs = [cr for cr in full if cr.backtest is not None]
    if len(runs) < 2:
        raise ValueError("PBO needs at least 2 successful candidates")
    first = runs[0].backtest
    assert first is not None
    dates = first.nav_dates
    if len(dates) != len(first.nav):
        raise ValueError(
            f"Candidate 0 has {len(dates)} nav_dates but {len(first.nav)} NAV values"
        )
    for i, cr in enumerate(runs[1:], start=1):
        assert cr.backtest is not None
        candidate_dates = cr.backtest.nav_dates
        if len(candidate_dates) != len(cr.backtest.nav):
            raise ValueError(
                f"Candidate {i} has {len(candidate_dates)} nav_dates but "
                f"{len(cr.backtest.nav)} NAV values"
            )
        if len(candidate_dates) != len(dates):
            raise ValueError("Candidate NAV dates do not match: length differs")
        for j, (expected, actual) in enumerate(
            zip(dates, candidate_dates, strict=True)
        ):
            if actual != expected:
                raise ValueError(
                    "Candidate NAV dates do not match "
                    f"at index {j}: expected {expected}, got {actual}"
                )
    if len(dates) < 2 * n_blocks:
        raise ValueError("Not enough data.")

    edges = np.linspace(0, len(dates), n_blocks + 1).astype(int)
    perf = np.zeros((n_blocks, len(runs)))
    for b in range(n_blocks):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi - lo < 2:
            raise ValueError("Not enough data.")
        start, end = dates[lo], dates[hi - 1]
        for j, cr in enumerate(runs):
            assert cr.backtest is not None
            rets = metrics.returns_from_nav(cr.backtest.window(start, end).nav)
            perf[b, j] = float(rets.mean()) if rets.size else 0.0
    return perf


def _stitch(
    returns: Sequence[NDArray[np.floating]],
    dates: list[datetime],
    folds: Sequence[Fold],
    initial_cash: float,
    periods_per_year: float,
    risk_free_rate: float,
) -> tuple[NDArray[np.floating], list[datetime], PerformanceSummary | None]:
    """Compound the per-fold OOS return segments into one continuous curve.

    Returns are stacked (not NAV levels) because each segment comes from a
    different selected policy's run; the trading metrics on the stitched summary
    are not meaningful (read them per fold) -- the NAV metrics are the headline.
    """
    if not returns:
        return np.array([], dtype=float), [], None
    stacked = np.concatenate(returns)
    nav = np.concatenate([[initial_cash], initial_cash * np.cumprod(1.0 + stacked)])
    nav_dates = [folds[0].test[0], *dates]
    stitched = BacktestResult(
        policy_name="walk_forward_oos",
        asset_order=[],
        nav_dates=nav_dates,
        nav=nav,
        periods=[],
    )
    summary = PerformanceSummary.from_backtest(
        stitched,
        active_only=False,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    return nav, nav_dates, summary
