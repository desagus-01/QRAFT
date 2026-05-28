from dataclasses import dataclass
from typing import Literal

from polars.dataframe.frame import DataFrame

from time_series.transforms.inverses import (
    InverseSpec,
)


@dataclass(frozen=True)
class TransformDecision:
    kind: Literal["none", "polynomial", "difference", "seasonality"]
    order: int | None = None


#
# @dataclass(frozen=True)
# class AppliedTransform:
#     asset: str
#     decision: TransformDecision
#     inverse_spec: InverseSpec | None = None
#


@dataclass(frozen=True)
class PipelineAssetBatchRes:
    type: Literal["trend", "seasonality"]
    decision: dict
    inverse_spec: dict[str, InverseSpec] | None
    updated_data: DataFrame
    all_tests: dict | None


@dataclass(frozen=True)
class UnivariatePreprocess:
    post_data: DataFrame
    inverse_specs: dict[str, list[InverseSpec]]
    needs_further_modelling: list[str]
