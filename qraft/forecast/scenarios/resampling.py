import numpy as np
from numpy import random
from numpy._typing import NDArray
from polars import DataFrame

from scenarios.types import ProbVector


def weighted_bootstrapping_idx(
    data: DataFrame,
    prob_vector: ProbVector,
    n_samples: int,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """
    Draw row indices from a weighted (probability) bootstrapping scheme.

    Parameters
    ----------
    data : DataFrame
        Polars DataFrame whose rows correspond to the probability vector.
    prob_vector : ProbVector
        List-like of non-negative weights that sum to one; length must equal
        ``data.height``.
    n_samples : int
        Number of indices to sample (with replacement).
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.int64]
        Integer indices sampled according to ``prob_vector`` with replacement.

    Raises
    ------
    ValueError
        If the length of ``prob_vector`` does not match the number of rows in ``data``.
    """
    if data.height != len(prob_vector):
        raise ValueError(
            f"Data size {data.height} and probability vector size {len(prob_vector)} do not match."
        )
    rng = random.default_rng(seed)
    return rng.choice(data.height, size=n_samples, replace=True, p=prob_vector)


def weighted_bootstrapping(
    data: DataFrame,
    prob_vector: ProbVector,
    n_samples: int,
    seed: int | None = None,
) -> DataFrame:
    """
    Return a resampled DataFrame using probability weights.

    Parameters
    ----------
    data : DataFrame
        DataFrame to sample rows from.
    prob_vector : ProbVector
        Probability weights corresponding to rows of ``data``.
    n_samples : int
        Number of rows to sample (with replacement).
    seed : int | None, optional
        RNG seed for reproducibility.

    Returns
    -------
    DataFrame
        Polars DataFrame formed by selecting rows according to sampled indices.
    """
    sample_row_n = weighted_bootstrapping_idx(
        data=data, prob_vector=prob_vector, n_samples=n_samples, seed=seed
    )
    return data[sample_row_n]
