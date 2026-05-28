import logging

import polars as pl
from polars.dataframe.frame import DataFrame

from policy import PreprocessConfig
from scenarios.types import ProbVector
from time_series.preprocessing.apply import (
    apply_deseason,
    apply_detrend,
    overwrite_with_transforms,
)
from time_series.preprocessing.decisions import (
    deseason_decision_rule,
    detrend_decision_rule,
)
from time_series.preprocessing.types import (
    PipelineAssetBatchRes,
    UnivariatePreprocess,
)
from time_series.preprocessing.white_noise import test_increments_idd
from time_series.selection.seasonality import (
    seasonality_diagnostic,
)
from time_series.selection.trend import trend_diagnostic
from time_series.transforms.inverses import InverseSpec
from utils.helpers import (
    get_assets_names,
)

logger = logging.getLogger(__name__)


def deseason_pipeline(
    data: DataFrame,
    assets: list[str] | None = None,
    include_diagnostics: bool = False,
) -> PipelineAssetBatchRes:
    """Diagnose, decide, and apply deterministic seasonal adjustment."""
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    diagnostics = seasonality_diagnostic(
        data=data,
        assets=assets,
    )

    decision = deseason_decision_rule(diagnostics)

    updated, inverse_specs = apply_deseason(
        data=data,
        decision=decision,
    )

    return PipelineAssetBatchRes(
        type="seasonality",
        decision=decision,
        inverse_spec=inverse_specs,
        updated_data=updated,
        all_tests=diagnostics if include_diagnostics else None,
    )


def detrend_pipeline(
    data: DataFrame,
    prob: ProbVector,
    assets: list[str] | None = None,
    cfg: PreprocessConfig | None = None,
    include_diagnostics: bool = False,
) -> PipelineAssetBatchRes:
    """Diagnose, decide, and apply detrending per asset."""
    if cfg is None:
        cfg = PreprocessConfig()
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    diagnostics = trend_diagnostic(
        data=data,
        assets=assets,
        prob=prob,
        order_max=cfg.trend_order_max,
        threshold_order=cfg.trend_threshold_order,
        trend_type=cfg.trend_type,
    )

    per_asset_decision = detrend_decision_rule(
        detrend_res=diagnostics,
        assets=assets,
    )

    updated, inverse_specs = apply_detrend(
        data=data,
        prob=prob,
        decision=per_asset_decision,
    )

    return PipelineAssetBatchRes(
        type="trend",
        decision=per_asset_decision,
        inverse_spec=inverse_specs,
        updated_data=updated,
        all_tests=diagnostics if include_diagnostics else None,
    )


# TODO: Review dropping nulls blankly - prob is a better way
def run_univariate_preprocess(
    data: pl.DataFrame,
    prob: ProbVector,
    assets: list[str] | None = None,
    cfg: PreprocessConfig | None = None,
) -> UnivariatePreprocess:
    """
    Pipeline:
      1) Screen assets by increments white-noise
      2) Detrend selected assets
      3) Deseason selected assets

    Returns
    -------
    UnivariatePreprocess
        post_data:
            Data after selected preprocessing steps have been applied.
        inverse_specs:
            Per-asset inverse transforms to restore forecasts later.
        needs_further_modelling:
            Assets whose increments failed the white-noise screen.
    """
    if cfg is None:
        cfg = PreprocessConfig()
    if assets is None:
        assets = get_assets_names(df=data, assets=assets)

    logger.info(
        "Starting univariate preprocess: rows=%d assets=%s",
        data.height,
        assets,
    )

    assets_need_preprocess = test_increments_idd(
        data=data,
        original_prob=prob,
        assets=assets,
        cfg=cfg.iid,
    )

    inverse_specs: dict[str, list[InverseSpec]] = {
        asset: [] for asset in assets_need_preprocess
    }

    # Trend
    detrend = detrend_pipeline(
        data=data.select(["date", *assets]),
        assets=assets_need_preprocess,
        include_diagnostics=False,
        prob=prob,
        cfg=cfg,
    )

    if detrend.inverse_spec is not None:
        for asset, spec in detrend.inverse_spec.items():
            inverse_specs[asset].append(spec)

    after_detrend = overwrite_with_transforms(
        base=data,
        patch=detrend.updated_data,
        assets=assets,
        suffix="_detrend",
    )

    # Seasonality
    deseason = deseason_pipeline(
        data=after_detrend.select(["date", *assets]),
        assets=assets_need_preprocess,
        include_diagnostics=False,
    )

    if deseason.inverse_spec is not None:
        for asset, spec in deseason.inverse_spec.items():
            inverse_specs[asset].append(spec)

    final = overwrite_with_transforms(
        base=after_detrend,
        patch=deseason.updated_data,
        assets=assets,
        suffix="_deseason",
    )

    logger.info("Finished univariate preprocess: inverse_specs=%s", inverse_specs)

    return UnivariatePreprocess(
        post_data=final,
        inverse_specs=inverse_specs,
        needs_further_modelling=assets_need_preprocess,
    )
