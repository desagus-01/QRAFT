from dataclasses import dataclass
from typing import Literal

from polars.dataframe.frame import DataFrame

from qraft.forecast.time_series.transforms.inverses import (
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

    @property
    def assets_already_iid(self) -> list[str]:
        all_assets = self.post_data.columns
        return [a for a in all_assets if a not in self.needs_further_modelling]

    def filter_to(self, keep: set[str]) -> "UnivariatePreprocess":
        cols = ["date"] + [c for c in self.post_data.columns if c in keep]
        filtered_data = self.post_data.select(cols)

        filtered_inverse = {k: v for k, v in self.inverse_specs.items() if k in keep}
        filtered_modelling = [a for a in self.needs_further_modelling if a in keep]
        return UnivariatePreprocess(
            post_data=filtered_data,
            inverse_specs=filtered_inverse,
            needs_further_modelling=filtered_modelling,
        )
