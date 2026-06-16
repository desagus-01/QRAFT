import logging

import polars as pl
from polars.dataframe.frame import DataFrame

from qraft.core.configs import IIDConfig
from qraft.core.panel import ScenarioPanel
from qraft.core.probability.prob_vector import ProbVector
from qraft.core.scenarios.copula_marginal import CopulaMarginalModel
from qraft.forecast.time_series.tests.iid import (
    PerAssetTestResult,
    TestResultByAsset,
    copula_lag_independence_test,
    ellipsoid_lag_test,
    univariate_arch_test,
)

logger = logging.getLogger(__name__)


def run_iid_simple(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags: int,
    arch_lags: tuple[int, ...],
) -> dict[str, TestResultByAsset]:
    """Cheaper iid screens: mean autocorrelation + second-moment (ARCH) clustering."""
    ellipsoid_test = ellipsoid_lag_test(
        data=data,
        prob=prob,
        lags=lags,
        assets=assets,
    )
    arch_effects_test = univariate_arch_test(
        data=data,
        lags=arch_lags,
        assets=assets,
    )
    return {
        "ellipsoid": ellipsoid_test,
        "arch": arch_effects_test,
    }


def run_iid_complex(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    lags: int,
    mc_iters: int,
    perm_test_iters: int,
    perm_test_min_iters: int,
    perm_test_batch_iters: int,
    perm_test_ci_level: float,
    significance_level: float,
    seed: int | None = None,
) -> dict[str, TestResultByAsset]:
    """Run the expensive copula IID screen."""
    copula_marginal_model = CopulaMarginalModel.from_data_and_prob(
        data=data.select(assets),
        prob=prob,
    )

    copula_lag_test_res = copula_lag_independence_test(
        copula=copula_marginal_model.copula_grades,
        prob=copula_marginal_model.prob,
        lags=lags,
        assets=assets,
        seed=seed,
        mc_iters=mc_iters,
        perm_test_iters=perm_test_iters,
        perm_test_min_iters=perm_test_min_iters,
        perm_test_batch_iters=perm_test_batch_iters,
        perm_test_ci_level=perm_test_ci_level,
        significance_level=significance_level,
    )

    return {
        "copula": copula_lag_test_res,
    }


def _ellipsoid_failed(
    simple_tests: dict[str, dict[str, PerAssetTestResult]],
    asset: str,
    rho_thresh: float,
) -> bool:
    """Only flag when a lag is both statistically AND practically significant."""
    return any(
        res.reject_null and abs(res.stat) >= rho_thresh
        for res in simple_tests["ellipsoid"][asset].results.values()
    )


def _arch_failed(
    simple_tests: dict[str, dict[str, PerAssetTestResult]], asset: str
) -> bool:
    return bool(simple_tests["arch"][asset].rejected)


def check_white_noise(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str],
    cfg: IIDConfig | None = None,
    seed: int | None = None,
) -> dict[str, bool]:
    """Return whether each asset passes the white-noise screen."""
    if cfg is None:
        cfg = IIDConfig()

    simple_tests = run_iid_simple(
        data=data,
        prob=prob,
        assets=assets,
        lags=cfg.lags_simple,
        arch_lags=cfg.arch_lags,
    )

    simple_pass = {
        asset: not (
            _ellipsoid_failed(simple_tests, asset, cfg.ellipsoid_rho_threshold)
            or _arch_failed(simple_tests, asset)
        )
        for asset in assets
    }

    assets_for_copula = [asset for asset, passed in simple_pass.items() if passed]

    logger.info(
        "White-noise simple screen: passed_simple=%s, sent_to_complex=%s",
        assets_for_copula,
        assets_for_copula,
    )

    if not assets_for_copula:
        logger.info("White-noise final screen: passed_all=[]")
        return simple_pass

    complex_tests = run_iid_complex(
        data=data,
        prob=prob,
        assets=assets_for_copula,
        lags=cfg.lags_complex,
        mc_iters=cfg.mc_iters,
        perm_test_iters=cfg.perm_test_iters,
        perm_test_min_iters=cfg.perm_test_min_iters,
        perm_test_batch_iters=cfg.perm_test_batch_iters,
        perm_test_ci_level=cfg.perm_test_ci_level,
        significance_level=cfg.significance_level,
        seed=seed,
    )

    copula_pass = {
        asset: not complex_tests["copula"][asset].rejected
        for asset in assets_for_copula
    }

    final_pass = simple_pass.copy()
    for asset in assets_for_copula:
        final_pass[asset] = final_pass[asset] and copula_pass[asset]

    logger.info(
        "White-noise final screen: passed_all=%s",
        [asset for asset, passed in final_pass.items() if passed],
    )

    return final_pass


def _find_nonwhite_noise_assets(
    increments: ScenarioPanel,
    assets: list[str],
    cfg: IIDConfig | None = None,
    seed: int | None = None,
) -> list[str]:
    """Return assets whose increments fail the white-noise screen."""
    wn = check_white_noise(
        data=increments.values.select(assets),
        assets=assets,
        prob=increments.prob,
        cfg=cfg,
        seed=seed,
    )
    return [a for a, ok in wn.items() if not ok]


def test_non_idd(
    data: pl.DataFrame,
    prob: ProbVector,
    assets: list[str],
    on_increment: bool = False,
    cfg: IIDConfig | None = None,
    seed: int | None = None,
) -> list[str]:
    """return those assets that are not iid."""
    panel = ScenarioPanel.from_log_prices(data.select(assets), prob)
    return _find_nonwhite_noise_assets(
        increments=panel.diff() if on_increment else panel,
        assets=assets,
        cfg=cfg,
        seed=seed,
    )
