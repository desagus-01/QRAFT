import logging
import time
from collections.abc import Sequence
from datetime import datetime
from functools import wraps
from typing import NamedTuple

import numpy as np
import polars as pl
import polars.selectors as cs
from matplotlib.axes import Axes
from numpy.typing import NDArray

from qraft.core.metrics import tail_cutoff
from qraft.core.probability.prob_vector import ProbVector

logger = logging.getLogger(__name__)


def str_to_datetime(date_time_str: str) -> datetime:
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_time_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"time data {date_time_str!r} is not ISO-8601")


class SplitDF(NamedTuple):
    first_half: pl.DataFrame
    second_half: pl.DataFrame


def split_df_in_half(data: pl.DataFrame) -> SplitDF:
    height = data.height

    if height % 2 != 0:
        height -= 1
        data = data.slice(0, height)

    mid = height // 2
    first_half = data.slice(0, mid)
    second_half = data.slice(mid, mid)

    return SplitDF(first_half, second_half)


def get_assets_names(df: pl.DataFrame, assets: list[str] | None = None) -> list[str]:
    if assets is None:
        return [c for c in df.columns if c != "date"]
    return df.select(assets).columns


def weighted_moments(
    data: NDArray[np.floating], weights: ProbVector
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    avg = np.average(data, axis=1, weights=weights)
    var = np.average((data.T - avg) ** 2, axis=0, weights=weights)
    return avg, np.sqrt(var)


def indicator_quantile_marginal(
    data: pl.DataFrame, target_quantile: float, prob: ProbVector | None = None
) -> pl.DataFrame:
    values = data.select(cs.numeric()).to_numpy()
    threshold = tail_cutoff(
        values,
        prob=prob,
        alpha=target_quantile,
        axis=0,
        distribution_type="pnl",
    )

    return data.with_columns(quant_ind=(cs.numeric() <= threshold).cast(pl.Int8))


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s took %.4fs", func.__name__, elapsed)
        return result

    return wrapper


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def selection_concentration(selected: Sequence[str]) -> float:
    if not selected:
        return 0.0
    counts = [selected.count(params) for params in set(selected)]
    return max(counts) / len(selected)


def most_common(selected: Sequence[str]) -> str:
    if not selected:
        return ""
    return max(set(selected), key=selected.count)


def plot_diagnostics_bar(
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


def column_or_nan(frame: pl.DataFrame, name: str, length: int) -> NDArray[np.floating]:
    if name not in frame.columns:
        return np.full(length, np.nan)
    return np.array(frame[name].to_list(), dtype=float)


def truncate_label(label: str, max_len: int = 42) -> str:
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 3]}..."


def path_nav(returns: Sequence[float] | NDArray[np.floating]) -> NDArray[np.floating]:
    returns_array = np.asarray(returns, dtype=float)
    if returns_array.size == 0:
        return np.array([], dtype=float)
    return np.concatenate(([1.0], np.cumprod(1.0 + returns_array)))


def format_optional_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def format_optional_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1%}"
