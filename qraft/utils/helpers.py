import logging
import time
from datetime import datetime
from functools import wraps
from typing import NamedTuple

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray

from qraft.core.metrics import tail_cutoff
from qraft.core.probability.prob_vector import ProbVector

logger = logging.getLogger(__name__)


def str_to_datetime(date_time_str: str) -> datetime:
    formats = [
        "%b %d %Y %I:%M%p",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y",
        "%d-%m-%Y %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_time_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"time data {date_time_str!r} does not match any known format")


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
