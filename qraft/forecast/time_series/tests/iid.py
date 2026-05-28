from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import polars as pl
import scipy.stats as st
from numpy.typing import NDArray
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from policy import IIDConfig
from scenarios.panel import compensate_prob
from scenarios.types import ProbVector
from time_series.estimation import weighted_covariance, weighted_mean
from time_series.tests.types import HypTestRes, format_hyp_test_result
from utils.helpers import (
    get_assets_names,
    split_df_in_half,
)

_DEFAULT_IID = IIDConfig()

StatFunc = Callable[[np.ndarray, ProbVector, np.ndarray], float]
PairTest = Callable[..., HypTestRes]
TestResultByAsset = dict[str, "PerAssetTestResult"]


@dataclass
class PerAssetTestResult:
    """Per-asset test results and rejected subtests."""

    results: dict[str, HypTestRes]
    rejected: list[str]

    @property
    def p_vals(self) -> list[float]:
        """Return p-values for all subtests."""
        return [res.p_val for res in self.results.values()]


@dataclass(frozen=True, slots=True)
class MeanCovRes:
    """Weighted mean and covariance result."""

    assets: list[str]
    means: NDArray[np.floating]
    cov: NDArray[np.floating]


def run_lagged_tests(
    data: pl.DataFrame,
    prob: ProbVector,
    assets: list[str] | None,
    test_fn: PairTest,
    lags: int = _DEFAULT_IID.lags_simple,
    **test_kwargs: Any,
) -> TestResultByAsset:
    """Run a pairwise test over lags for each selected asset."""
    sel_assets = get_assets_names(data, assets)
    results: TestResultByAsset = {}

    for asset in sel_assets:
        x = data.get_column(asset).to_numpy()
        per_lag: dict[str, HypTestRes] = {}

        for lag in range(1, lags + 1):
            if lag >= x.shape[0]:
                raise ValueError(
                    f"Lag {lag} is too large for asset '{asset}' with "
                    f"{x.shape[0]} observations."
                )

            pair_np = np.empty((x.shape[0] - lag, 2), dtype=float)
            pair_np[:, 0] = x[lag:]
            pair_np[:, 1] = x[:-lag]

            lag_prob = compensate_prob(prob=prob, n_remove=lag)

            res = test_fn(
                pair_np,
                lag_prob,
                assets=(asset, f"{asset}_lag_{lag}"),
                **test_kwargs,
            )
            per_lag[f"lag_{lag}"] = res

        rejected_lags = [k for k, res in per_lag.items() if res.reject_null]

        results[asset] = PerAssetTestResult(
            results=per_lag,
            rejected=rejected_lags,
        )

    return results


def _sample_meancov_np(
    data_np: np.ndarray,
    prob: ProbVector,
    assets: list[str] | tuple[str, str],
) -> MeanCovRes:
    """Compute weighted mean and covariance for a 2D array."""
    if data_np.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data_np.shape}")

    prob = prob.reshape(-1)
    if data_np.shape[0] != prob.shape[0]:
        raise ValueError(
            f"Probability vector length {prob.shape[0]} does not match "
            f"number of observations {data_np.shape[0]}"
        )

    means = weighted_mean(data_np, prob)
    cov = weighted_covariance(data_np, prob, center=True)

    return MeanCovRes(
        assets=list(assets),
        means=means,
        cov=cov,
    )


def autocorrelation_pair_test(
    pair_np: np.ndarray,
    prob: ProbVector,
    assets: tuple[str, str],
) -> HypTestRes:
    """Test linear dependence between a series and one lag."""
    cov = _sample_meancov_np(pair_np, prob, assets).cov
    var_t = cov[0, 0]
    var_lag = cov[1, 1]
    cov_t_lag = cov[0, 1]

    denom = np.sqrt(var_t * var_lag)
    corr = 0.0 if np.isclose(denom, 0.0) else float(cov_t_lag / denom)

    test_statistic = abs(corr) * np.sqrt(pair_np.shape[0])
    p_val = float(2 * (1 - st.norm.cdf(test_statistic)))

    return format_hyp_test_result(stat=corr, p_val=p_val, null="Independence")


def ellipsoid_lag_test(
    data: pl.DataFrame,
    prob: ProbVector,
    lags: int = _DEFAULT_IID.lags_simple,
    assets: list[str] | None = None,
) -> TestResultByAsset:
    """Run the ellipsoid lag test for each asset and lag."""
    return run_lagged_tests(
        data=data,
        prob=prob,
        lags=lags,
        assets=assets,
        test_fn=autocorrelation_pair_test,
    )


def _copula_eval(
    pobs: np.ndarray,
    p: ProbVector,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate the empirical copula at given points."""
    less_eq = pobs[:, None, :] <= points[None, :, :]
    inside = np.all(less_eq, axis=2)
    return p @ inside


def _copula_eval_2d(
    pobs: np.ndarray,
    p: ProbVector,
    points: np.ndarray,
) -> np.ndarray:
    """Fast empirical copula evaluation for the 2D case."""
    inside = (pobs[:, 0, None] <= points[None, :, 0]) & (
        pobs[:, 1, None] <= points[None, :, 1]
    )
    return p @ inside


def _sw_stat(
    pobs: np.ndarray,
    p: ProbVector,
    u_points: np.ndarray,
) -> float:
    """Compute the Schweizer-Wolff statistic on a fixed grid."""
    if pobs.shape[1] == 2:
        est = _copula_eval_2d(pobs, p, u_points)
    else:
        est = _copula_eval(pobs, p, u_points)

    indep = u_points.prod(axis=1)
    return 12.0 * np.abs(est - indep).mean()


def _draw_mc_points(
    dim: int,
    rng: np.random.Generator,
    mc_iters: int = _DEFAULT_IID.mc_iters,
) -> np.ndarray:
    """Draw uniform Monte Carlo points once for a test."""
    return rng.uniform(0.0, 1.0, size=(mc_iters, dim))


def sw_mc(
    pobs: np.ndarray,
    p: ProbVector,
    u_points: np.ndarray,
) -> float:
    """Compute the SW statistic using pre-drawn Monte Carlo points."""
    return _sw_stat(pobs, p, u_points)


def independence_permutation_test(
    pair_np: np.ndarray,
    prob: ProbVector,
    assets: tuple[str, str],
    stat_fun: StatFunc = sw_mc,
    iter: int = _DEFAULT_IID.perm_test_iters,
    rng: np.random.Generator | None = None,
) -> HypTestRes:
    """Run a permutation independence test on a 2-column array."""
    if rng is None:
        rng = np.random.default_rng()

    arr = np.asarray(pair_np, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected pair array of shape (n_obs, 2), got {arr.shape}")

    u_points = _draw_mc_points(dim=arr.shape[1], rng=rng)
    stat = stat_fun(arr, prob, u_points)

    n = arr.shape[0]
    null_dist = np.empty(iter, dtype=float)

    permuted = arr.copy()
    first_col = arr[:, 0]

    for i in range(iter):
        permuted[:, 0] = first_col[rng.permutation(n)]
        null_dist[i] = stat_fun(permuted, prob, u_points)

    p_val = (1.0 + np.count_nonzero(null_dist >= stat)) / (iter + 1.0)

    return format_hyp_test_result(stat=stat, p_val=p_val, null="Independence")


def copula_lag_independence_test(
    copula: pl.DataFrame,
    prob: ProbVector,
    lags: int = _DEFAULT_IID.lags_complex,
    assets: list[str] | None = None,
) -> TestResultByAsset:
    """Run the copula lag independence test for each asset and lag."""
    return run_lagged_tests(
        data=copula,
        prob=prob,
        lags=lags,
        assets=assets,
        test_fn=independence_permutation_test,
    )


def kolmogrov_smirnov_2stest(
    dist_1: NDArray[np.floating],
    dist_2: NDArray[np.floating],
) -> HypTestRes:
    """Run a two-sample KS test."""
    res = st.kstest(
        dist_1,
        dist_2,
        alternative="two-sided",
    )

    return format_hyp_test_result(
        stat=res.statistic,
        p_val=res.pvalue,
        null="Identically Distributed",
    )


def univariate_kolmogrov_smirnov_test(
    data: pl.DataFrame,
    assets: list[str] | None = None,
) -> TestResultByAsset:
    """Compare the two halves of each asset with a KS test."""
    sel_assets = get_assets_names(data, assets)
    out: TestResultByAsset = {}

    for asset in sel_assets:
        split = split_df_in_half(data.select(asset))
        split_1 = split.first_half.to_numpy().ravel()
        split_2 = split.second_half.to_numpy().ravel()

        res = kolmogrov_smirnov_2stest(split_1, split_2)

        out[asset] = PerAssetTestResult(
            results={"split": res},
            rejected=(["split"] if res.reject_null else []),
        )

    return out


def ljung_box_test(
    data: pl.DataFrame | NDArray[np.floating],
    lags: tuple[int, ...],
    asset: str | None = None,
    degrees_of_freedom: int = 0,
) -> PerAssetTestResult:
    """Run Ljung-Box tests on a 1D series."""
    if isinstance(data, pl.DataFrame):
        if asset is not None:
            array = data.get_column(asset).to_numpy()
        else:
            if data.width != 1:
                raise ValueError(
                    "asset must be provided when data has multiple columns."
                )
            array = data.to_series(0).to_numpy()
    else:
        array = np.asarray(data)
        if array.ndim != 1:
            raise ValueError(
                f"data must be 1D when not a Polars DataFrame; got ndim={array.ndim}"
            )

    lb = acorr_ljungbox(
        x=array,
        lags=lags,
        boxpierce=False,
        model_df=degrees_of_freedom,
    )

    per_lag: dict[str, HypTestRes] = {}
    for lag in lb.index:
        per_lag[f"lag_{int(lag)}"] = format_hyp_test_result(
            stat=lb.loc[lag, "lb_stat"],
            p_val=lb.loc[lag, "lb_pvalue"],
            null=f"No autocorrelation up to lag {int(lag)}",
        )

    rejected = [k for k, res in per_lag.items() if res.reject_null]
    return PerAssetTestResult(results=per_lag, rejected=rejected)


def arch_test(
    residual_array: NDArray[np.floating],
    lags_to_test: tuple[int, ...],
    degrees_of_freedom: int,
) -> PerAssetTestResult:
    """Run ARCH tests over the requested lags."""
    arch_lag_res: dict[str, HypTestRes] = {}

    for lag in lags_to_test:
        arch_res = het_arch(residual_array, nlags=lag, ddof=degrees_of_freedom)
        arch_lag_res[f"{lag}"] = format_hyp_test_result(
            stat=arch_res[0],
            p_val=arch_res[1],
            null=f"no ARCH effects up to lag {int(lag)}",
        )

    rejected = [k for k, res in arch_lag_res.items() if res.reject_null]
    return PerAssetTestResult(results=arch_lag_res, rejected=rejected)
