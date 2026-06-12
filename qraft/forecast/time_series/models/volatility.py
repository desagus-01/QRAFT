import logging
import re
from itertools import product

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from numpy._typing import NDArray

from qraft.core.configs import VolatilityModelConfig
from qraft.forecast.time_series.models.fitted_types import (
    GARCH_DISTRIBUTIONS,
    AutoGARCHRes,
)

logger = logging.getLogger(__name__)

GARCH_FIT_SCALE = 100.0


def _garch_base_model(
    asset_array: NDArray[np.floating],
    innovation_distribution: GARCH_DISTRIBUTIONS = "t",
    p_order: int = 1,
    o_order: int = 0,
    q_order: int = 1,
) -> ARCHModelResult:
    base_model = arch_model(
        y=asset_array * GARCH_FIT_SCALE,
        mean="zero",
        p=p_order,
        o=o_order,
        q=q_order,
        dist=innovation_distribution,
        rescale=False,
    ).fit(disp="off")
    return base_model


def _descale_garch_params(params: dict[str, float]) -> dict[str, float]:
    descaled = {k: v for k, v in params.items()}
    if "omega" in descaled:
        descaled["omega"] = descaled["omega"] / (GARCH_FIT_SCALE**2)
    return descaled


def _garch_result(
    model: ARCHModelResult,
    *,
    model_order: tuple[int, int, int],
    admissible: bool,
    persistence: float,
    fallback_reason: str | None = None,
) -> AutoGARCHRes:
    params = _descale_garch_params(model.params.to_dict())  # type: ignore[attr-defined]
    return AutoGARCHRes(
        model_order=model_order,
        degrees_of_freedom=len(model.params),
        criteria="bic",
        criteria_res=model.bic,
        params=params,
        p_values=model.pvalues,  # type: ignore[attr-defined]
        invariants=model.std_resid,  # type: ignore[attr-defined]
        residuals=model.resid / GARCH_FIT_SCALE,  # type: ignore[attr-defined]
        conditional_volatility=model.conditional_volatility / GARCH_FIT_SCALE,  # type: ignore[attr-defined]
        persistence=persistence,
        admissible=admissible,
        fallback_reason=fallback_reason,
    )


def _garch_persistence_calc(params: dict[str, float]) -> float:
    alpha_sum = sum(v for k, v in params.items() if k.startswith("alpha"))
    beta_sum = sum(v for k, v in params.items() if k.startswith("beta"))
    gamma_sum = sum(v for k, v in params.items() if k.startswith("gamma"))
    return alpha_sum + beta_sum + 0.5 * gamma_sum


def _garch_boundaries_check(
    params: dict[str, float],
    cfg: VolatilityModelConfig,
) -> bool:
    vals = {k: v for k, v in params.items()}

    def _lag_num(name: str) -> int:
        m = re.search(r"\[(\d+)\]$", name)
        return int(m.group(1)) if m else 1

    for k, v in vals.items():
        if (
            (k.startswith("alpha") or k.startswith("beta") or k.startswith("gamma"))
            and (_lag_num(k) > 1)
            and abs(v) < cfg.tolerance_zero
        ):
            return True

    beta_items = sorted(
        [(k, v) for k, v in vals.items() if k.startswith("beta")],
        key=lambda kv: _lag_num(kv[0]),
    )
    if len(beta_items) >= 2:
        betas = [v for _, v in beta_items]
        for i in range(len(betas)):
            for j in range(i + 1, len(betas)):
                if abs(betas[i] - betas[j]) < cfg.tolerance_dups:
                    return True

    return False


def _admissable_garch_model(
    params: dict[str, float], cfg: VolatilityModelConfig
) -> bool:
    persistence = _garch_persistence_calc(params)

    if persistence >= cfg.max_persistence:
        return False
    if _garch_boundaries_check(params, cfg):
        return False

    omega = params.get("omega", 0.0)
    if omega <= 0:
        return False

    return True


def auto_garch(
    asset_array: NDArray[np.floating],
    cfg: VolatilityModelConfig | None = None,
    asset_name: str | None = None,
) -> list[AutoGARCHRes]:
    if cfg is None:
        cfg = VolatilityModelConfig()

    base_model = _garch_base_model(asset_array=asset_array)
    if base_model.convergence_flag != 0:
        raise ValueError(
            f"Baseline GARCH(1,0,1) failed to converge with flag={base_model.convergence_flag}"
        )

    garch_candidates: list[AutoGARCHRes] = []

    for p, q, o, distribution in product(
        range(1, cfg.max_p_order + 1),
        range(1, cfg.max_q_order + 1),
        range(0, cfg.max_o_order + 1),
        cfg.candidate_distributions,
    ):
        key = (p, o, q)
        if (key == (1, 0, 1)) and (distribution == "t"):
            continue

        proposed_model = arch_model(
            asset_array * GARCH_FIT_SCALE,
            mean="zero",
            p=p,
            o=o,
            q=q,
            dist=distribution,
            rescale=False,
        ).fit(disp="off")

        if proposed_model.convergence_flag != 0:
            logger.info(
                "Asset=%s GARCH%s dist=%s did not converge (flag=%s); dropping candidate",
                asset_name,
                key,
                distribution,
                proposed_model.convergence_flag,
            )
            continue

        params = _descale_garch_params(proposed_model.params.to_dict())  # type: ignore[attr-defined]
        persistence = _garch_persistence_calc(params)
        if not _admissable_garch_model(params, cfg):
            logger.info(
                "Asset=%s GARCH%s dist=%s failed admissibility checks; dropping candidate",
                asset_name,
                key,
                distribution,
            )
            continue

        garch_candidates.append(
            _garch_result(
                proposed_model,
                model_order=key,
                persistence=persistence,
                admissible=True,
            )
        )

    base_params = _descale_garch_params(base_model.params.to_dict())  # type: ignore[attr-defined]
    base_persistence = _garch_persistence_calc(base_params)
    base_admissible = _admissable_garch_model(base_params, cfg)
    base_res = _garch_result(
        base_model,
        model_order=(1, 0, 1),
        persistence=base_persistence,
        admissible=base_admissible,
        fallback_reason=None
        if base_admissible
        else "baseline_failed_admissibility_checks",
    )

    if base_admissible:
        garch_candidates.append(base_res)
    else:
        logger.warning(
            "Asset=%s Baseline GARCH(1, 0, 1) failed admissibility checks; keeping it as a last-resort fallback",
            asset_name,
        )

    if not garch_candidates:
        logger.warning(
            "Asset=%s No admissible GARCH candidates were fitted; returning baseline GARCH(1, 0, 1) as fallback",
            asset_name,
        )
        garch_candidates.append(base_res)

    return sorted(garch_candidates, key=lambda res: res.criteria_res)
