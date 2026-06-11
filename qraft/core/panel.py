from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame

from qraft.core.probability.distributions import uniform_probs
from qraft.core.probability.prob_vector import ProbVector, validate_prob_vector


def redistribute_prob_mass(
    prob: ProbVector,
    dropped_idx: NDArray[np.int_],
) -> ProbVector:
    if dropped_idx.size == 0:
        return prob.copy()

    kept = np.delete(prob, dropped_idx)
    if kept.size == 0:
        raise ValueError("Cannot redistribute: all entries would be removed")

    mass_removed = float(prob[dropped_idx].sum())
    return kept + mass_removed * (kept / kept.sum())


def compensate_prob(prob: ProbVector, n_remove: int) -> ProbVector:
    if n_remove < 0:
        raise ValueError("n_remove must be non-negative")
    return redistribute_prob_mass(prob, np.arange(n_remove, dtype=np.int_))


@dataclass(frozen=True)
class ScenarioPanel:
    values: pl.DataFrame
    dates: pl.Series | None  # TODO: Change to just time to make simulations use it too
    prob: ProbVector

    def __post_init__(self) -> None:
        if "date" in self.values.columns:
            raise ValueError("ScenarioPanel.values must not contain a 'date' column; ")
        if self.values.height == 0:
            raise ValueError("ScenarioPanel cannot be empty")
        try:
            values_are_finite = np.isfinite(self.values.to_numpy()).all()
        except TypeError as exc:
            raise ValueError(
                "ScenarioPanel.values must contain only finite numeric values"
            ) from exc
        if not values_are_finite:
            raise ValueError(
                "ScenarioPanel.values must contain only finite numeric values"
            )

        if self.prob.shape[0] != self.values.height:
            raise ValueError(
                f"prob length {self.prob.shape[0]} != rows {self.values.height}"
            )
        validate_prob_vector(self.prob)
        self.prob.setflags(write=False)
        object.__setattr__(self, "prob", self.prob)

        if self.dates is not None:
            if len(self.dates) != self.values.height:
                raise ValueError(
                    f"dates length {len(self.dates)} != rows {self.values.height}"
                )
            if self.dates.name != "date":
                object.__setattr__(self, "dates", self.dates.rename("date"))

    @classmethod
    def from_log_prices(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
        drop_nulls: bool = False,
    ) -> ScenarioPanel:
        if "date" in df.columns:
            dates: pl.Series | None = df.get_column("date")
            values = df.drop("date")
        else:
            dates = None
            values = df

        if prob is None:
            prob = uniform_probs(values.height)

        if drop_nulls:
            values, dates, prob = cls._drop_null_rows(values, dates, prob)

        return cls(values=values, dates=dates, prob=prob)

    @classmethod
    def from_prices(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
    ) -> ScenarioPanel:
        dates = df.get_column("date") if "date" in df.columns else None
        values = df.drop("date") if "date" in df.columns else df
        arr = values.to_numpy()
        try:
            if not np.isfinite(arr).all():
                raise ValueError("from_prices: values contain NaN/inf.")
            if (arr <= 0).any():
                raise ValueError("from_prices: prices must be strictly positive.")
        except TypeError as exc:
            raise ValueError("from_prices: prices must be numeric.") from exc
        log_values = values.select(pl.col(c).log() for c in values.columns)
        panel_prob = prob if prob is not None else uniform_probs(values.height)
        return cls(values=log_values, dates=dates, prob=panel_prob)

    def to_frame(self) -> DataFrame:
        if self.dates is None:
            return self.values

        return DataFrame({"date": self.dates}).hstack(self.values)

    def drop_nulls(self) -> ScenarioPanel:
        values, dates, prob = self._drop_null_rows(self.values, self.dates, self.prob)
        if values is self.values:
            return self

        return ScenarioPanel(values=values, dates=dates, prob=prob)

    @staticmethod
    def _drop_null_rows(
        values: pl.DataFrame,
        dates: pl.Series | None,
        prob: ProbVector,
    ) -> tuple[pl.DataFrame, pl.Series | None, ProbVector]:
        if values.height != prob.shape[0]:
            raise ValueError(f"prob length {prob.shape[0]} != rows {values.height}")
        if dates is not None and len(dates) != values.height:
            raise ValueError(f"dates length {len(dates)} != rows {values.height}")

        null_mask = values.select(pl.any_horizontal(pl.all().is_null())).to_series()

        if not null_mask.any():
            return values, dates, prob

        keep_mask = ~null_mask
        clean = values.filter(keep_mask)
        new_dates = dates.filter(keep_mask) if dates is not None else None

        dropped_idx = np.flatnonzero(null_mask.to_numpy())
        new_prob = redistribute_prob_mass(prob, dropped_idx)

        return clean, new_dates, new_prob

    def diff(self, lag: int = 1) -> ScenarioPanel:
        if lag < 1:
            raise ValueError("lag must be >= 1")

        diffed = self.values.with_columns(
            [pl.col(c).diff(lag).alias(c) for c in self.values.columns]
        ).slice(lag)
        new_dates = self.dates.slice(lag) if self.dates is not None else None
        new_prob = compensate_prob(self.prob, lag)
        return ScenarioPanel(values=diffed, dates=new_dates, prob=new_prob)

    def with_prob(self, prob: ProbVector) -> ScenarioPanel:
        return ScenarioPanel(values=self.values, dates=self.dates, prob=prob)

    def map_values_same_rows(self, values: pl.DataFrame) -> ScenarioPanel:
        if values.height != self.values.height:
            raise ValueError(
                f"map_values_same_rows requires identical row count; "
                f"got {values.height} vs {self.values.height}. "
                "Use filter_rows() or diff() / drop_nulls() for row-changing ops."
            )
        return ScenarioPanel(values=values, dates=self.dates, prob=self.prob)

    def filter_rows(self, mask: pl.Series) -> ScenarioPanel:
        if len(mask) != self.values.height:
            raise ValueError(f"mask length {len(mask)} != rows {self.values.height}")
        if not mask.any():
            raise ValueError("filter_rows would remove every row")

        new_values = self.values.filter(mask)
        new_dates = self.dates.filter(mask) if self.dates is not None else None
        dropped_idx = np.flatnonzero((~mask).to_numpy())
        new_prob = redistribute_prob_mass(self.prob, dropped_idx)
        return ScenarioPanel(values=new_values, dates=new_dates, prob=new_prob)

    @property
    def has_dates(self) -> bool:
        return self.dates is not None

    def require_dates(self) -> pl.Series:
        if self.dates is None:
            raise ValueError(
                "This operation requires a dated panel, but AssetPanel.dates is None."
            )
        return self.dates

    @property
    def asset_names(self) -> list[str]:
        return self.values.columns

    @property
    def n_rows(self) -> int:
        return self.values.height

    def __len__(self) -> int:
        return self.values.height
