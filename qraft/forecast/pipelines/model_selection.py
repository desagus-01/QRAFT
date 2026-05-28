import logging
from typing import Mapping

import numpy as np
from numpy._typing import NDArray
from polars import DataFrame

from policy import MeanModelConfig, PipelineConfig, VolatilityModelConfig
from time_series.models.fitted_types import (
    AutoARMARes,
    AutoGARCHRes,
    DemeanRes,
    MeanModelRes,
    UnivariateRes,
)
from time_series.models.mean import (
    auto_arma,
)
from time_series.models.model_quality import (
    SelectionAudit,
    score_audit,
)
from time_series.models.volatility import (
    auto_garch,
)
from time_series.tests.iid import arch_test, ljung_box_test
from time_series.tests.multiple import multiple_tests_rejected

logger = logging.getLogger(__name__)


def by_criteria(res: AutoARMARes | AutoGARCHRes) -> float:
    return res.criteria_res


def _describe_mean_model(res: MeanModelRes | None) -> str:
    if res is None:
        return "random_walk"

    if isinstance(res, DemeanRes):
        return "demean"

    if isinstance(res, AutoARMARes):
        return f"ARMA{res.model_order}"

    return type(res).__name__


def _describe_vol_model(res: AutoGARCHRes | None) -> str:
    if res is None:
        return "none"

    return f"GARCH{res.model_order}"


def describe_univariate_result(res: UnivariateRes) -> str:
    mean_label = _describe_mean_model(res.mean_res)
    vol_label = _describe_vol_model(res.volatility_res)
    return f"mean={mean_label}, vol={vol_label}"


def asset_needs_volatility_model(
    residual: NDArray[np.floating],
    degrees_of_freedom: int,
    cfg: VolatilityModelConfig | None = None,
) -> bool:
    """
    Decide whether a volatility (GARCH) model is required for an asset.

    The decision is based on Ljung–Box tests run on squared residuals
    (a proxy for conditional heteroskedasticity) and ARCH tests on
    residuals. If the number of rejected tests exceeds the thresholds in
    ``cfg`` a volatility model is recommended.
    """
    if cfg is None:
        cfg = VolatilityModelConfig()
    residual_sq = residual**2
    ljung_box_pvals = ljung_box_test(
        residual_sq, lags=cfg.ljung_box_lags, degrees_of_freedom=degrees_of_freedom
    ).p_vals
    arch_vals = arch_test(
        residual, lags_to_test=cfg.arch_lags, degrees_of_freedom=degrees_of_freedom
    ).p_vals

    ljung_box_rejected = multiple_tests_rejected(ljung_box_pvals)
    arch_rejected = multiple_tests_rejected(arch_vals)
    return (sum(ljung_box_rejected) >= cfg.min_ljung_box_rejections) or (
        sum(arch_rejected) >= cfg.min_arch_rejections
    )


def asset_needs_mean_modelling(
    data: NDArray[np.floating],
    degrees_of_freedom: int,
    cfg: MeanModelConfig | None = None,
) -> bool:
    """
    Determine whether an asset series requires mean modelling (ARMA).
    """
    if cfg is None:
        cfg = MeanModelConfig()
    ljung_box = ljung_box_test(
        data=data, lags=cfg.ljung_box_lags, degrees_of_freedom=degrees_of_freedom
    )
    ljung_box_rejected = multiple_tests_rejected(ljung_box.p_vals)
    return sum(ljung_box_rejected) >= cfg.min_ljung_box_rejections


def needs_volatility_modelling(
    mean_model_res: Mapping[str, MeanModelRes],
    cfg: VolatilityModelConfig | None = None,
) -> list[str]:
    """
    Determine which assets require volatility modelling based on residual tests.
    """
    if cfg is None:
        cfg = VolatilityModelConfig()
    needs: list[str] = []

    for asset, res in mean_model_res.items():
        if asset_needs_volatility_model(
            residual=res.residuals,
            degrees_of_freedom=res.degrees_of_freedom,
            cfg=cfg,
        ):
            needs.append(asset)

    return needs


def needs_mean_modelling(
    data: DataFrame,
    assets_to_test: list[str],
    degrees_of_freedom: int,
    cfg: MeanModelConfig | None = None,
) -> list[str]:
    """
    Identifies assets with significant autocorrelation in series using the Ljung–Box test.
    """
    if cfg is None:
        cfg = MeanModelConfig()
    result = []
    for asset in assets_to_test:
        array = data.select(asset).drop_nulls().to_numpy().ravel()
        if asset_needs_mean_modelling(
            array,
            cfg=cfg,
            degrees_of_freedom=degrees_of_freedom,
        ):
            result.append(asset)
    return result


def _demean_fallback(asset_array: NDArray[np.floating]) -> DemeanRes:
    mean = float(np.mean(asset_array))
    residuals = asset_array - mean
    scale = float(np.std(residuals, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return DemeanRes(
        model_order=None,
        degrees_of_freedom=0,
        params={"mean": mean},
        residuals=residuals,
        residual_scale=scale,
    )


def _select_first_candidate_with_valid_residuals(
    candidates_by_information_criteria: list[AutoARMARes],
    cfg: MeanModelConfig,
) -> AutoARMARes | None:
    for model in candidates_by_information_criteria:
        if not asset_needs_mean_modelling(
            model.residuals, degrees_of_freedom=model.degrees_of_freedom, cfg=cfg
        ):
            return model
    return None


def get_appropriate_mean_model(
    asset_array: NDArray[np.floating],
    asset_name: str,
    cfg: MeanModelConfig | None = None,
    audit: SelectionAudit | None = None,
) -> MeanModelRes:
    """
    Select the best ARMA mean model satisfying residual diagnostics.
    """
    if cfg is None:
        cfg = MeanModelConfig()
    try:
        candidate_models_res = auto_arma(
            asset_array=asset_array,
            cfg=cfg,
        )
    except ValueError:
        logger.warning(
            "Asset=%s no admissible ARMA models found; falling back to demean",
            asset_name,
        )
        if audit is not None:
            audit.add_event(
                "MEAN_FALLBACK_DEMEAN", f"Asset={asset_name} no admissible ARMA models"
            )
        return _demean_fallback(asset_array)

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)
    model = _select_first_candidate_with_valid_residuals(
        candidates_by_information_criteria, cfg=cfg
    )
    if model is not None:
        return model

    logger.warning(
        "Asset=%s no ARMA candidate passed residual diagnostics; falling back to best %s model",
        asset_name,
        cfg.information_criteria.upper(),
    )
    if audit is not None:
        audit.add_event(
            "MEAN_FALLBACK_BEST_IC_NO_DIAG_PASS",
            f"Asset={asset_name} no ARMA candidate passed residual diagnostics; fallback to best {cfg.information_criteria.upper()}",
        )
    return min(candidate_models_res, key=by_criteria)


def _select_model_with_valid_arch_tests(
    candidates_by_information_criteria: list[AutoGARCHRes],
    cfg: VolatilityModelConfig,
) -> AutoGARCHRes | None:
    for model in candidates_by_information_criteria:
        invariants = np.asarray(model.invariants, dtype=float)
        if not np.all(np.isfinite(invariants)):
            continue

        if not asset_needs_volatility_model(
            residual=invariants,
            degrees_of_freedom=model.degrees_of_freedom,
            cfg=cfg,
        ):
            return model
    return None


def run_best_garch(
    asset_array: NDArray[np.floating],
    asset_name: str,
    cfg: VolatilityModelConfig | None = None,
    audit: SelectionAudit | None = None,
) -> AutoGARCHRes | None:
    """
    Select a GARCH model from candidate fits that passes residual diagnostics.
    """
    if cfg is None:
        cfg = VolatilityModelConfig()
    try:
        candidate_models_res = auto_garch(
            asset_array=asset_array,
            cfg=cfg,
        )
    except ValueError as exc:
        logger.warning(
            "Asset=%s no usable GARCH candidates were fitted; keeping volatility model as none (%s)",
            asset_name,
            exc,
        )
        return None

    candidates_by_information_criteria = sorted(candidate_models_res, key=by_criteria)
    model = _select_model_with_valid_arch_tests(
        candidates_by_information_criteria, cfg=cfg
    )
    if model is not None:
        return model

    logger.warning(
        "Asset=%s no GARCH candidate passed residual diagnostics; falling back to best information-criterion model",
        asset_name,
    )
    if audit is not None:
        audit.add_event(
            "VOL_FALLBACK_BEST_IC_NO_DIAG_PASS",
            f"Asset={asset_name} no GARCH candidate passed residual diagnostics; fallback to best information-criterion",
        )

    return min(candidate_models_res, key=by_criteria)


def mean_modelling_pipeline(
    data: DataFrame,
    assets: list[str],
    cfg: MeanModelConfig | None = None,
) -> tuple[dict[str, MeanModelRes], dict[str, SelectionAudit]]:
    """
    Return a mean-modelling result for every asset, plus a selection audit per asset.
    """
    if cfg is None:
        cfg = MeanModelConfig()
    assets_needing_arma = needs_mean_modelling(
        data=data, assets_to_test=assets, degrees_of_freedom=0, cfg=cfg
    )

    asset_mean_model_res: dict[str, MeanModelRes] = {}
    mean_audits: dict[str, SelectionAudit] = {}
    for asset in assets:
        array = data.select(asset).drop_nulls().to_numpy().ravel()

        audit = SelectionAudit(events=[], notes=[])
        if asset in assets_needing_arma:
            res = get_appropriate_mean_model(
                array, asset_name=asset, cfg=cfg, audit=audit
            )
            asset_mean_model_res[asset] = res
            logger.info(
                "Selected mean model for %s: %s", asset, _describe_mean_model(res)
            )
        else:
            mean = array.mean().item()
            residuals = array - mean
            res = DemeanRes(
                model_order=None,
                degrees_of_freedom=0,
                params={"mean": mean},
                residuals=residuals,
                residual_scale=float(np.std(residuals, ddof=1)),
            )
            asset_mean_model_res[asset] = res
            logger.info(
                "Selected mean model for %s: %s", asset, _describe_mean_model(res)
            )

        mean_audits[asset] = audit

    return asset_mean_model_res, mean_audits


def volatility_modelling_pipeline(
    mean_model_res: Mapping[str, MeanModelRes],
    cfg: VolatilityModelConfig | None = None,
) -> tuple[dict[str, AutoGARCHRes], dict[str, SelectionAudit]]:
    """
    Fit volatility (GARCH) models for assets that require them and return audits.
    """
    if cfg is None:
        cfg = VolatilityModelConfig()
    assets_needing_garch = needs_volatility_modelling(mean_model_res, cfg=cfg)
    logger.info("Assets needing volatility modelling: %s", assets_needing_garch)

    asset_vol_model_res: dict[str, AutoGARCHRes] = {}
    vol_audits: dict[str, SelectionAudit] = {}

    for asset in assets_needing_garch:
        audit = SelectionAudit(events=[], notes=[])
        res = run_best_garch(
            mean_model_res[asset].residuals, asset, cfg=cfg, audit=audit
        )
        if res is not None:
            asset_vol_model_res[asset] = res
        vol_audits[asset] = audit
        logger.info(
            "Selected volatility model for %s: %s", asset, _describe_vol_model(res)
        )

    return asset_vol_model_res, vol_audits


def get_univariate_results(
    data: DataFrame,
    assets_to_model: list[str],
    cfg: PipelineConfig | None = None,
) -> dict[str, UnivariateRes]:
    """
    Run mean and volatility modelling pipelines and aggregate results per asset, including quality.
    """
    if cfg is None:
        from policy import DEFAULT_PIPELINE_CONFIG

        cfg = DEFAULT_PIPELINE_CONFIG

    mean_modelling, mean_audits = mean_modelling_pipeline(
        data=data, assets=assets_to_model, cfg=cfg.mean
    )
    volatility_modelling, vol_audits = volatility_modelling_pipeline(
        mean_model_res=mean_modelling, cfg=cfg.volatility
    )

    all_assets = [c for c in data.columns if c != "date"]
    asset_model: dict[str, UnivariateRes] = {}

    for asset in all_assets:
        combined_audit = SelectionAudit(events=[], notes=[])
        if asset in mean_audits:
            combined_audit.events.extend(mean_audits[asset].events)
            combined_audit.notes.extend(mean_audits[asset].notes)
        if asset in vol_audits:
            combined_audit.events.extend(vol_audits[asset].events)
            combined_audit.notes.extend(vol_audits[asset].notes)

        quality = score_audit(combined_audit, cfg.quality)
        asset_model[asset] = UnivariateRes(
            mean_res=mean_modelling.get(asset),
            volatility_res=volatility_modelling.get(asset),
            quality=quality,
        )
        logger.info(
            "Final univariate result for %s: %s",
            asset,
            describe_univariate_result(asset_model[asset]),
        )

    return asset_model
