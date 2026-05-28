from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Self

import polars as pl
from numpy import interp
from polars import DataFrame

from probability.sampling import marginal_quantile_mapping, sample_copula
from scenarios.panel import ScenarioPanel
from scenarios.types import ProbVector, validate_prob_vector


def _compute_cdf_and_pobs(
    data: pl.DataFrame,
    marginal_name: str,
    prob: ProbVector,
    compute_pobs: bool = True,
) -> pl.DataFrame:
    """Compute empirical CDF (and optionally pseudo-observations) for one marginal.

    Returns a DataFrame with columns ``[index, <marginal_name>, prob, cdf]``
    and, when ``compute_pobs`` is True, also ``pobs`` aligned to the original
    row order. Input must be null-free.
    """
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
        """Raise if marginals / cdfs / copula differ in columns or height,
        or if any frame height disagrees with ``len(self.prob)``.
        """
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
        """Construct a CopulaMarginalModel from an :class:`AssetPanel`.

        The panel must be null-free; callers should run ``panel.drop_nulls()``
        first when appropriate.
        """
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
        """Construct from a raw DataFrame and optional prior.

        Delegates to :meth:`from_panel` after normalising via
        :class:`AssetPanel`.
        """
        return cls.from_panel(ScenarioPanel.from_frame(data, prob))

    def to_panel(self) -> ScenarioPanel:
        """Convert back to an :class:`AssetPanel` by interpolating marginals.

        Each marginal is reconstructed by mapping copula pseudo-observations
        through the empirical CDF.
        """
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
        seed: int | None = None,
        target_marginals: dict[str, Literal["t", "norm"]] | None = None,
        *,
        target_copula: Literal["t", "norm"] | None = None,
        copula_fit_method: Literal["ml", "irho", "itau"] | None = None,
    ) -> ScenarioPanel:
        """Apply CMA updates to marginals and/or the copula and return a panel.

        Parameters
        ----------
        seed : int | None, optional
            RNG seed forwarded to copula resampling if requested.
        target_marginals : dict[str, {"t", "norm"}] | None, optional
            Per-asset target marginal families for quantile mapping.
        target_copula : {"t", "norm"} | None, optional
            Target parametric copula family. If provided, the copula will be
            refit/resampled using ``copula_fit_method``.
        copula_fit_method : {"ml", "irho", "itau"} | None, optional
            Copula fitting method; if omitted defaults may be used by
            underlying utilities.

        Returns
        -------
        AssetPanel
            Reconstructed scenarios with dates (if any) and probability vector
            carried through.

        Raises
        ------
        ValueError
            If neither ``target_copula`` nor ``target_marginals`` is provided.
        """
        if target_copula is None and target_marginals is None:
            raise ValueError("Choose a target marginal or target copula!")
        if target_copula is not None and copula_fit_method is None:
            raise ValueError(
                "You must choose a copula fit method if you have selected a copula!"
            )

        model = self

        if (target_copula is not None) and (copula_fit_method is not None):
            model = model.update_copula(
                seed=seed, target_copula=target_copula, fit_method=copula_fit_method
            )

        if target_marginals is not None:
            model = model.update_marginals(target_marginals)

        return model.to_panel()
