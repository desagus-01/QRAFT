from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Self

import polars as pl
from core.panel import ScenarioPanel
from core.probability.sampling import marginal_quantile_mapping, sample_copula
from core.probability.prob_vector import ProbVector, validate_prob_vector
from numpy import interp
from polars import DataFrame


def _compute_cdf_and_pobs(
    data: pl.DataFrame,
    marginal_name: str,
    prob: ProbVector,
    compute_pobs: bool = True,
) -> pl.DataFrame:
    if data.null_count().sum_horizontal().item() > 0:
        raise ValueError(
            "compute_cdf_and_pobs expects null-free data; drop nulls first."
        )
    df = (
        data.select(pl.col(marginal_name))
        .with_row_index()
        .with_columns(prob=prob)
        .sort(marginal_name)
        .with_columns(
            cdf=pl.cum_sum("prob") * data.height / (data.height + 1),
        )
    )

    if compute_pobs:
        df = df.with_columns(
            pobs=pl.col("cdf").gather(pl.col("index").arg_sort()),
        )

    return df


@dataclass(frozen=True)
class CMAConfig:
    """How to reshape the joint distribution under copula-marginal adjustment."""

    target_copula: Literal["t", "norm"] | None = None
    target_marginals: dict[str, Literal["t", "norm"]] | None = None
    copula_fit_method: Literal["ml", "irho", "itau"] = "itau"

    def __post_init__(self) -> None:
        if self.target_copula is None and self.target_marginals is None:
            raise ValueError(
                "CMAConfig needs at least one of target_copula or target_marginals."
            )


# TODO: Need to find a way to allow nulls at the start
@dataclass(frozen=True)
class CopulaMarginalModel:
    marginals: DataFrame
    cdfs: DataFrame
    copula_grades: DataFrame
    prob: ProbVector
    dates: pl.Series | None

    def __post_init__(self) -> None:
        self._validate_frames()
        validate_prob_vector(self.prob)
        self.prob.setflags(write=False)
        if self.dates is not None and len(self.dates) != self.marginals.height:
            raise ValueError(
                f"dates length {len(self.dates)} != frame height {self.marginals.height}"
            )

    def _validate_frames(self) -> None:
        frames = {
            "marginals": self.marginals,
            "cdfs": self.cdfs,
            "copula": self.copula_grades,
        }
        ref_name, ref = "marginals", self.marginals
        for name, frame in frames.items():
            if name == ref_name:
                continue
            if frame.columns != ref.columns:
                raise ValueError(
                    f"column mismatch: {ref_name}.columns={ref.columns} "
                    f"vs {name}.columns={frame.columns}"
                )
            if frame.height != ref.height:
                raise ValueError(
                    f"height mismatch: {ref_name}.height={ref.height} "
                    f"vs {name}.height={frame.height}"
                )
        if ref.height != len(self.prob):
            raise ValueError(f"frame height {ref.height} != len(prob) {len(self.prob)}")

    @classmethod
    def from_panel(cls, panel: ScenarioPanel) -> CopulaMarginalModel:
        cdf_cols: dict[str, pl.Series] = {}
        copula_cols: dict[str, pl.Series] = {}
        sorted_marginals: dict[str, pl.Series] = {}

        for col in panel.values.iter_columns():
            asset = col.name
            temp = _compute_cdf_and_pobs(panel.values, asset, panel.prob)

            cdf_cols[asset] = temp["cdf"]
            copula_cols[asset] = temp["pobs"]
            sorted_marginals[asset] = temp[asset]

        return cls(
            marginals=DataFrame(sorted_marginals),
            cdfs=DataFrame(cdf_cols),
            copula_grades=DataFrame(copula_cols),
            prob=panel.prob,
            dates=panel.dates,
        )

    @classmethod
    def from_data_and_prob(
        cls, data: DataFrame, prob: ProbVector | None = None
    ) -> CopulaMarginalModel:
        return cls.from_panel(ScenarioPanel.from_frame(data, prob))

    def to_panel(self) -> ScenarioPanel:
        interp_res = {}
        for asset in self.marginals.columns:
            interp_res[asset] = interp(
                x=self.copula_grades.select(asset).to_numpy().ravel(),
                xp=self.cdfs.select(asset).to_numpy().ravel(),
                fp=self.marginals.select(asset).to_numpy().ravel(),
            )
        return ScenarioPanel(
            values=DataFrame(interp_res),
            dates=self.dates,
            prob=self.prob,
        )

    def update_marginals(self, target_dists: dict[str, Literal["t", "norm"]]) -> Self:
        """Replace selected marginals with target families via quantile mapping.

        For each specified marginal the function treats the current copula
        pseudo-observations as grades and maps the empirical marginal
        quantiles to the target distribution (Student-t or Normal).
        """
        new_marginals = self.marginals
        new_cdfs = self.cdfs

        for marginal, target_dist in target_dists.items():
            grades = self.copula_grades.select(marginal).to_numpy().ravel()
            sample_values = self.marginals.select(marginal).to_numpy().ravel()

            new_scenarios = marginal_quantile_mapping(
                marginal=sample_values, grades=grades, kind=target_dist
            )

            rebuilt = _compute_cdf_and_pobs(
                pl.DataFrame({marginal: new_scenarios.ravel()}),
                marginal,
                self.prob,
                compute_pobs=False,
            )

            new_marginals = new_marginals.with_columns(
                rebuilt[marginal].alias(marginal)
            )
            new_cdfs = new_cdfs.with_columns(rebuilt["cdf"].alias(marginal))

        return replace(self, marginals=new_marginals, cdfs=new_cdfs)

    def update_copula(
        self,
        seed: int | None = None,
        fit_method: Literal["ml", "irho", "itau"] = "itau",
        target_copula: Literal["t", "norm"] = "t",
    ) -> Self:
        """Re-fit or sample a copula given the current pseudo-observations."""
        new_copula = sample_copula(
            self.copula_grades,
            seed=seed,
            parametric_copula=target_copula,
            fit_method=fit_method,
        )
        return replace(self, copula_grades=new_copula)

    def update_distribution(
        self,
        cfg: CMAConfig,
        seed: int | None = None,
    ) -> ScenarioPanel:
        """Apply CMA updates (marginals and/or copula) and return a panel."""
        model: CopulaMarginalModel = self
        if cfg.target_copula is not None:
            model = model.update_copula(
                seed=seed,
                target_copula=cfg.target_copula,
                fit_method=cfg.copula_fit_method,
            )
        if cfg.target_marginals is not None:
            model = model.update_marginals(cfg.target_marginals)
        return model.to_panel()
