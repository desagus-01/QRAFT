import math
from dataclasses import dataclass
from typing import TypedDict

from policy import IIDConfig

_DEFAULT_ROUNDING = 4
_DEFAULT_IID = IIDConfig()


@dataclass(frozen=True, slots=True)
class HypTestRes:
    stat: float
    p_val: float
    sign_lvl: float
    null: str
    reject_null: bool
    desc: str


class HypTestConclusion(TypedDict):
    reject_null: bool
    desc: str


def hyp_test_conclusion(
    p_val: float, null_hyp: str, sign_level: float
) -> HypTestConclusion:
    if p_val >= sign_level:
        return {
            "reject_null": False,
            "desc": (
                f"Fail to reject null hypothesis of {null_hyp} at {sign_level} significance."
            ),
        }
    else:
        return {
            "reject_null": True,
            "desc": f"Reject null hypothesis of {null_hyp} at {sign_level} significance.",
        }


def format_hyp_test_result(
    stat: float,
    p_val: float,
    null: str = "Independence",
    sign_level: float = _DEFAULT_IID.significance_level,
) -> HypTestRes:
    try:
        stat = float(stat)
        p_val = float(p_val)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid stat or p_val: stat={stat}, p_val={p_val}")

    if math.isnan(stat) or math.isnan(p_val):
        raise ValueError(f"Invalid stat or p_val: stat={stat}, p_val={p_val}")

    hyp_conc = hyp_test_conclusion(p_val, null_hyp=null, sign_level=sign_level)

    return HypTestRes(
        stat=round(stat, _DEFAULT_ROUNDING),
        p_val=round(p_val, _DEFAULT_ROUNDING),
        sign_lvl=sign_level,
        null=null,
        reject_null=hyp_conc["reject_null"],
        desc=hyp_conc["desc"],
    )
