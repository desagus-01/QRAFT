import logging
import re
from itertools import product

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from numpy._typing import NDArray

from policy import VolatilityModelConfig
from time_series.models.fitted_types import GARCH_DISTRIBUTIONS, AutoGARCHRes

logger = logging.getLogger(__name__)


def _garch_base_model(
    asset_array: NDArray[np.floating],
    innovation_distribution: GARCH_DISTRIBUTIONS = "t",
    p_order: int = 1,
    o_order: int = 0,
    q_order: int = 1,
) -> ARCHModelResult:
    base_model = arch_model(
        y=asset_array,
        mean="zero",
        p=p_order,
        o=o_order,
        q=q_order,
        dist=innovation_distribution,
        rescale=False,
    ).fit(disp="off")
    return base_model


def _garch_persistence_calc(params: dict[str, float]) -> float:
    alpha_sum = sum(v for k, v in params.items() if k.startswith("alpha"))
    beta_sum = sum(v for k, v in params.items() if k.startswith("beta"))
    gamma_sum = sum(v for k, v in params.items() if k.startswith("gamma"))
    return alpha_sum + beta_sum + 0.5 * gamma_sum


def _garch_boundaries_check(
    params: dict[str, float],
    cfg: VolatilityModelConfig,
) -> bool:
    vals = {k: float(v) for k, v in params.items()}

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
            asset_array, mean="zero", p=p, o=o, q=q, dist=distribution, rescale=False
        ).fit(disp="off")

        if proposed_model.convergence_flag != 0:
            logger.info(
                "GARCH%s dist=%s did not converge (flag=%s); dropping candidate",
                key,
                distribution,
                proposed_model.convergence_flag,
            )
            continue

        params = proposed_model.params.to_dict()  # type: ignore[attr-defined]
        if not _admissable_garch_model(params, cfg):
            logger.info(
                "GARCH%s dist=%s failed admissibility checks; dropping candidate",
                key,
                distribution,
            )
            continue

        garch_candidates.append(
            AutoGARCHRes(
                model_order=key,
                degrees_of_freedom=len(proposed_model.params),
                criteria="bic",
                criteria_res=proposed_model.bic,
                params=params,
                p_values=proposed_model.pvalues,  # type: ignore[attr-defined]
                invariants=proposed_model.std_resid,  # type: ignore[attr-defined]
                residuals=proposed_model.resid,  # type: ignore[attr-defined]
                conditional_volatility=proposed_model.conditional_volatility,  # type: ignore[attr-defined]
            )
        )

    base_params = base_model.params.to_dict()  # type: ignore[attr-defined]
    base_res = AutoGARCHRes(
        model_order=(1, 0, 1),
        degrees_of_freedom=len(base_model.params),
        criteria="bic",
        criteria_res=base_model.bic,
        params=base_params,
        p_values=base_model.pvalues,  # type: ignore[attr-defined]
        residuals=base_model.resid,  # type: ignore[attr-defined]
        invariants=base_model.std_resid,  # type: ignore[attr-defined]
        conditional_volatility=base_model.conditional_volatility,  # type: ignore[attr-defined]
    )

    if _admissable_garch_model(base_params, cfg):
        garch_candidates.append(base_res)
    else:
        logger.warning(
            "Baseline GARCH(1, 0, 1) failed admissibility checks; keeping it as a last-resort fallback"
        )

    if not garch_candidates:
        logger.warning(
            "No admissible GARCH candidates were fitted; returning baseline GARCH(1, 0, 1) as fallback"
        )
        garch_candidates.append(base_res)

    return sorted(garch_candidates, key=lambda res: res.criteria_res)
