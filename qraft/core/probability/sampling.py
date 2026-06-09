import warnings
from typing import Literal

import numpy as np
import polars as pl
from copulae import NormalCopula, StudentCopula
from numpy._typing import NDArray
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
        Fitting method; falls back to ML if analytical method fails.
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
        if fit_method == "ml":
            raise RuntimeError(f"ML fit failed: {e}") from e

        warnings.warn(
            f"Fit method '{fit_method}' failed ({e!r}); falling back to 'ml'.",
            RuntimeWarning,
        )
        fit = cop.fit(values, method="ml", to_pobs=to_pobs)

    samples = fit.random(n=copula.height, seed=seed)

    return pl.DataFrame(samples, schema=copula.columns)
