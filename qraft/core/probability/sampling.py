from typing import Literal

import numpy as np
import polars as pl
from copulae import NormalCopula, StudentCopula
from qraft.core.probability.prob_vector import ProbVector
from numpy import random
from numpy._typing import NDArray
from polars import DataFrame
from scipy.stats import norm, t


def sample_marginal(
    data: pl.DataFrame, marginals: str, kind: Literal["t", "norm"] = "t"
) -> pl.DataFrame:
    values = data.select(marginals).to_numpy().flatten()

    if kind == "t":
        params = t.fit(values)
        frozen = t(*params)
    elif kind == "norm":
        params = norm.fit(values)
        frozen = norm(*params)
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")

    return pl.DataFrame({marginals: frozen.rvs(size=data.height)})


def marginal_quantile_mapping(
    marginal: NDArray[np.floating],
    grades: NDArray[np.floating],
    kind: Literal["t", "norm"] = "t",
) -> NDArray[np.floating]:
    if kind == "t":
        params = t.fit(marginal)
        frozen = t(*params)
    elif kind == "norm":
        params = norm.fit(marginal)
        frozen = norm(*params)
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")
    return frozen.ppf(grades)


def sample_copula(
    copula: pl.DataFrame,
    seed: int | None = None,
    parametric_copula: Literal["t", "norm"] = "t",
    fit_method: Literal["ml", "irho", "itau"] = "ml",
    to_pobs: bool = False,
) -> pl.DataFrame:
    """
    Fit a parametric copula to `copula` data and sample the same number of rows.

    Parameters
    ----------
    copula : pl.DataFrame
        Input data (typically uniform margins if to_pobs=False).
    seed : int | None
        Random seed for reproducibility.
    parametric_copula : {"t", "norm"}
        Which copula family to fit.
    fit_method : {"ml", "irho", "itau"}
        Fitting method.
    to_pobs : bool
        Whether to transform input values to pseudo-observations during fitting.

    Returns
    -------
    pl.DataFrame
        Samples with same shape and column names as input.
    """
    if copula.is_empty():
        raise ValueError("`copula` is empty; cannot fit/sample.")

    values = copula.to_numpy()
    n_dim = values.shape[1]

    cop_cls = {"t": StudentCopula, "norm": NormalCopula}.get(parametric_copula)
    if cop_cls is None:
        raise ValueError("parametric_copula must be one of {'t', 'norm'}")

    cop = cop_cls(n_dim)

    try:
        fit = cop.fit(values, method=fit_method, to_pobs=to_pobs)
    except Exception as e:
        raise RuntimeError(
            f"Copula fit failed for {parametric_copula!r} copula with "
            f"method {fit_method!r}: {e}"
        ) from e

    samples = fit.random(n=copula.height, seed=seed)

    return pl.DataFrame(samples, schema=copula.columns)


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
