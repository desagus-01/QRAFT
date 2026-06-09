import logging
import time
from functools import wraps
from typing import NamedTuple

import numpy as np
import polars as pl
import polars.selectors as cs
from qraft.core.probability.prob_vector import ProbVector
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
    """
    Retrieve asset column names from a DataFrame.

    If ``assets`` is provided the function validates and returns that list
    (raising if columns are missing by virtue of Polars selection semantics).
    If ``assets`` is ``None`` the function returns all columns except 'date'.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing asset columns (and possibly a 'date' column).
    assets : list[str] | None, optional
        Explicit list of asset column names to validate and return.

    Returns
    -------
    list[str]
        Asset column names.
    """
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
    data: pl.DataFrame, target_quantile: float
) -> pl.DataFrame:
    threshold = np.quantile(data, target_quantile)

    return data.with_columns(quant_ind=(cs.numeric() <= threshold).cast(pl.Int8))


def build_diff_df(data: pl.DataFrame, asset: str, diffs: int = 1) -> pl.DataFrame:
    if diffs <= 0:
        raise ValueError("Diffs need to be bigger than 0 dummy")
    return data.select(
        asset,
        *[
            pl.col(asset).diff(i).alias(f"{asset}_diff_{i}")
            for i in range(1, diffs + 1)
        ],
    )


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s took %.4fs", func.__name__, elapsed)
        return result

    return wrapper


def wide_to_long(data: pl.DataFrame, assets: list[str]) -> pl.DataFrame:
    return data.unpivot(
        index="date",
        on=list(assets),
        variable_name="ticker",
        value_name="adj_close",
    ).with_columns(pl.col("date").cast(pl.Date))
