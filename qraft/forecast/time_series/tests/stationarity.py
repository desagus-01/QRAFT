from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple

import polars as pl
from statsmodels.tsa.stattools import adfuller as sm_adfuller
from statsmodels.tsa.stattools import kpss as sm_kpss

from qraft.core.estimation import EquationTypes
from qraft.forecast.time_series.tests.types import (
    HypTestRes,
    format_hyp_test_result,
)

InferenceLabel = Literal[
    "stationary",
    "trend-stationary",
    "unit-root/non-stationary",
    "inconclusive/conflict",
]


class StationaryTestsRes(NamedTuple):
    test_stat: float
    p_val: float
    std_error: float | None


@dataclass(frozen=True, slots=True)
class StationarityInference:
    label: InferenceLabel
    sign_lvl: float
    details: str
    adf_c: HypTestRes
    adf_ct: HypTestRes
    kpss_level: HypTestRes
    kpss_trend: HypTestRes


@dataclass(frozen=True)
class Rule:
    label: InferenceLabel
    when: Callable[[bool, bool, bool, bool], bool]
    details: str


RULES: tuple[Rule, ...] = (
    Rule(
        "stationary",
        lambda A_c, A_ct, L, T: A_c and (not L),
        "ADF(c) rejects unit root; KPSS(level) fails to reject level stationarity.",
    ),
    Rule(
        "trend-stationary",
        lambda A_c, A_ct, L, T: A_ct and (not T),
        "ADF(ct) rejects unit root; KPSS(trend) fails to reject trend stationarity.",
    ),
    Rule(
        "unit-root/non-stationary",
        lambda A_c, A_ct, L, T: (not A_c) and L,
        "ADF(c) fails to reject unit root; KPSS(level) rejects level stationarity.",
    ),
)


ADF_REGRESSION_TYPES: dict[EquationTypes, Literal["n", "c", "ct", "ctt"]] = {
    "nc": "n",
    "c": "c",
    "ct": "ct",
    "ctt": "ctt",
}

KPSS_REGRESSION_TYPES: dict[Literal["trend", "level"], Literal["ct", "c"]] = {
    "trend": "ct",
    "level": "c",
}


def infer_stationarity(
    adf_c: HypTestRes,
    adf_ct: HypTestRes,
    kpss_level: HypTestRes,
    kpss_trend: HypTestRes,
) -> StationarityInference:
    A_c = adf_c.reject_null
    A_ct = adf_ct.reject_null
    L = kpss_level.reject_null
    T = kpss_trend.reject_null

    for r in RULES:
        if r.when(A_c, A_ct, L, T):
            return StationarityInference(
                label=r.label,
                sign_lvl=adf_c.sign_lvl,
                details=r.details,
                adf_c=adf_c,
                adf_ct=adf_ct,
                kpss_level=kpss_level,
                kpss_trend=kpss_trend,
            )

    return StationarityInference(
        label="inconclusive/conflict",
        sign_lvl=adf_c.sign_lvl,
        details="No clean ADF/KPSS agreement pattern at this significance level.",
        adf_c=adf_c,
        adf_ct=adf_ct,
        kpss_level=kpss_level,
        kpss_trend=kpss_trend,
    )


def augmented_dickey_fuller(
    data: pl.DataFrame,
    asset: str,
    metric: Literal["aic", "bic"] = "aic",
    lags: int | None = None,
    eq_type: EquationTypes = "nc",
) -> StationaryTestsRes:
    test_stat, p_val, *_ = sm_adfuller(
        data.select(asset).to_series().to_numpy(),
        maxlag=lags,
        regression=ADF_REGRESSION_TYPES[eq_type],
        autolag=metric.upper(),
    )

    return StationaryTestsRes(
        test_stat=float(test_stat),
        std_error=None,
        p_val=float(p_val),
    )


def kpss(
    data: pl.DataFrame, asset: str, null_type_stationarity: Literal["trend", "level"]
) -> StationaryTestsRes:
    test_stat, p_val, *_ = sm_kpss(
        data.select(asset).to_series().to_numpy(),
        regression=KPSS_REGRESSION_TYPES[null_type_stationarity],
        nlags="auto",
    )

    return StationaryTestsRes(
        test_stat=test_stat,
        p_val=p_val,
        std_error=None,
    )


def augmented_dickey_fuller_test(
    data: pl.DataFrame,
    asset: str,
    metric: Literal["aic", "bic"] = "aic",
    lags: int | None = None,
    eq_type: EquationTypes = "nc",
) -> HypTestRes:
    adf_res = augmented_dickey_fuller(
        data=data, asset=asset, metric=metric, lags=lags, eq_type=eq_type
    )

    return format_hyp_test_result(
        stat=adf_res.test_stat,
        p_val=adf_res.p_val,
        null="Unit Root/Non-Stationarity",
    )


def kpss_test(
    data: pl.DataFrame, asset: str, null_type_stationarity: Literal["trend", "level"]
) -> HypTestRes:
    kpss_res = kpss(
        data=data, asset=asset, null_type_stationarity=null_type_stationarity
    )

    return format_hyp_test_result(
        stat=kpss_res.test_stat,
        p_val=kpss_res.p_val,
        null=f"{null_type_stationarity} stationarity",
    )


def stationarity_tests(
    data: pl.DataFrame,
    asset: str,
    adf_metric: Literal["aic", "bic"] = "aic",
    lags: int | None = None,
) -> StationarityInference:
    adf_c = augmented_dickey_fuller_test(
        data=data, asset=asset, metric=adf_metric, lags=lags, eq_type="c"
    )
    adf_ct = augmented_dickey_fuller_test(
        data=data, asset=asset, metric=adf_metric, lags=lags, eq_type="ct"
    )

    kpss_res_lvl = kpss_test(data=data, asset=asset, null_type_stationarity="level")

    kpss_res_trend = kpss_test(data=data, asset=asset, null_type_stationarity="trend")

    return infer_stationarity(
        adf_c=adf_c, adf_ct=adf_ct, kpss_level=kpss_res_lvl, kpss_trend=kpss_res_trend
    )
