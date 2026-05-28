from dataclasses import dataclass
from typing import Literal

from polars import DataFrame

from scenarios.types import ProbVector
from time_series.tests.stationarity import (
    StationarityInference,
    stationarity_tests,
)
from time_series.transforms.detrend import (
    CandidateBatch,
    TrendCandidate,
    build_difference_candidates,
    build_polynomial_candidates,
)

TrendTypes = Literal["deterministic", "stochastic"]


@dataclass(frozen=True)
class TrendCandidateEvaluation:
    candidate: TrendCandidate
    stationarity_inf: StationarityInference

    @property
    def is_stationary(self) -> bool:
        return self.stationarity_inf.label == "stationary"


@dataclass(frozen=True)
class TrendSelection:
    asset: str
    trend_type: TrendTypes
    lowest_order_threshold: int
    selected: TrendCandidate | None
    evaluations: list[TrendCandidateEvaluation]

    @property
    def transformation_order_needed(self) -> int | None:
        return None if self.selected is None else self.selected.order

    @property
    def transformation_needed(self) -> bool:
        return (
            self.selected is not None
            and self.selected.order <= self.lowest_order_threshold
        )

    @property
    def description(self) -> str:
        if self.selected is None:
            return (
                f"No stationary transform found up to order "
                f"{self.lowest_order_threshold} for {self.trend_type}."
            )

        if self.transformation_needed:
            return (
                f"Transformation needed: {self.trend_type} "
                f"({self.selected.transform_type}) of order "
                f"{self.selected.order}."
            )

        return (
            f"Stationarity achieved at order {self.selected.order}, "
            f"but exceeds acceptance threshold "
            f"(tested up to {self.lowest_order_threshold})."
        )


@dataclass(frozen=True)
class AssetTrendDiagnostic:
    asset: str
    deterministic: TrendSelection | None = None
    stochastic: TrendSelection | None = None


def evaluate_candidates(
    batch: CandidateBatch,
) -> dict[str, list[TrendCandidateEvaluation]]:
    out: dict[str, list[TrendCandidateEvaluation]] = {}

    for asset, candidates in batch.candidates_by_asset.items():
        evaluations: list[TrendCandidateEvaluation] = []

        for candidate in candidates:
            if candidate.column_name not in batch.data.columns:
                raise ValueError(
                    f"Column {candidate.column_name} not found in candidate batch data."
                )

            stationarity_inference = stationarity_tests(
                batch.data, candidate.column_name
            )

            evaluations.append(
                TrendCandidateEvaluation(
                    candidate=candidate,
                    stationarity_inf=stationarity_inference,
                )
            )

        out[asset] = evaluations

    return out


def select_lowest_stationary_candidate(
    asset: str,
    trend_type: TrendTypes,
    evaluations: list[TrendCandidateEvaluation],
    threshold_order: int,
) -> TrendSelection:
    stationary_candidates_by_order = [
        evaluation
        for evaluation in sorted(evaluations, key=lambda x: x.candidate.order)
        if evaluation.is_stationary
    ]

    selected = (
        None
        if not stationary_candidates_by_order
        else stationary_candidates_by_order[0].candidate
    )

    return TrendSelection(
        asset=asset,
        trend_type=trend_type,
        lowest_order_threshold=threshold_order,
        selected=selected,
        evaluations=evaluations,
    )


def run_asset_trend_selection(
    batch: CandidateBatch,
    trend_type: TrendTypes,
    threshold_order: int,
) -> dict[str, TrendSelection]:
    evaluated = evaluate_candidates(batch=batch)

    return {
        asset: select_lowest_stationary_candidate(
            asset=asset,
            trend_type=trend_type,
            evaluations=evaluations,
            threshold_order=threshold_order,
        )
        for asset, evaluations in evaluated.items()
    }


def trend_diagnostic(
    data: DataFrame,
    assets: list[str],
    prob: ProbVector,
    order_max: int,
    threshold_order: int,
    *,
    trend_type: Literal["deterministic", "stochastic", "both"],
) -> dict[str, AssetTrendDiagnostic]:
    out = {asset: AssetTrendDiagnostic(asset=asset) for asset in assets}

    if trend_type in ("deterministic", "both"):
        deterministic_batch = build_polynomial_candidates(
            data=data,
            prob=prob,
            assets=assets,
            max_order=order_max,
        )
        deterministic_selection = run_asset_trend_selection(
            batch=deterministic_batch,
            trend_type="deterministic",
            threshold_order=threshold_order,
        )
        for asset, selection in deterministic_selection.items():
            out[asset] = AssetTrendDiagnostic(
                asset=asset,
                deterministic=selection,
                stochastic=out[asset].stochastic,
            )

    if trend_type in ("stochastic", "both"):
        stochastic_batch = build_difference_candidates(
            data=data,
            assets=assets,
            max_order=order_max,
        )
        stochastic_selection = run_asset_trend_selection(
            batch=stochastic_batch,
            trend_type="stochastic",
            threshold_order=threshold_order,
        )
        for asset, selection in stochastic_selection.items():
            out[asset] = AssetTrendDiagnostic(
                asset=asset,
                deterministic=out[asset].deterministic,
                stochastic=selection,
            )

    return out
