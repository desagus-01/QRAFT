from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame

from qraft.core.probability.distributions import uniform_probs
from qraft.core.probability.prob_vector import ProbVector, validate_prob_vector

ScenarioPanelKind = Literal["log_price", "return", "invariant", "level"]
DatetimeSeries = Annotated[pl.Series, "dtype: pl.Datetime"]


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
    dates: DatetimeSeries
    prob: ProbVector
    kind: ScenarioPanelKind = "level"

    def __post_init__(self) -> None:
        if self.kind not in ("log_price", "return", "invariant", "level"):
            raise ValueError(f"Unknown ScenarioPanel kind: {self.kind!r}")
        if "date" in self.values.columns:
            raise ValueError("ScenarioPanel.values must not contain a 'date' column; ")
        if self.values.height == 0:
            raise ValueError("ScenarioPanel cannot be empty")
        try:
            arr = self.values.to_numpy()
            finite_mask = np.isfinite(arr)
            if not finite_mask.all():
                bad_cols = [
                    col
                    for i, col in enumerate(self.values.columns)
                    if not np.isfinite(arr[:, i]).all()
                ]
                raise ValueError(
                    f"ScenarioPanel.values must contain only finite numeric "
                    f"values; columns with NaN/Inf: {bad_cols}"
                )
        except TypeError as exc:
            raise ValueError(
                "ScenarioPanel.values must contain only finite numeric values"
            ) from exc

        if self.prob.shape[0] != self.values.height:
            raise ValueError(
                f"prob length {self.prob.shape[0]} != rows {self.values.height}"
            )
        validate_prob_vector(self.prob)
        self.prob.setflags(write=False)
        object.__setattr__(self, "prob", self.prob)

        if len(self.dates) != self.values.height:
            raise ValueError(
                f"dates length {len(self.dates)} != rows {self.values.height}"
            )
        object.__setattr__(self, "dates", normalize_datetime_series(self.dates))

    @classmethod
    def from_log_prices(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
        drop_nulls: bool = False,
    ) -> ScenarioPanel:
        dates, values = _split_dates_and_values(df)

        if prob is None:
            prob = uniform_probs(values.height)

        if drop_nulls:
            values, dates, prob = cls._drop_null_rows(values, dates, prob)

        return cls(values=values, dates=dates, prob=prob, kind="log_price")

    @classmethod
    def from_prices(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
    ) -> ScenarioPanel:
        dates, values = _split_dates_and_values(df)
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
        return cls(values=log_values, dates=dates, prob=panel_prob, kind="log_price")

    @classmethod
    def from_returns(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
        drop_nulls: bool = False,
    ) -> ScenarioPanel:
        return cls._from_tagged_values(
            df=df, prob=prob, drop_nulls=drop_nulls, kind="return"
        )

    @classmethod
    def from_invariants(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
        drop_nulls: bool = False,
    ) -> ScenarioPanel:
        return cls._from_tagged_values(
            df=df, prob=prob, drop_nulls=drop_nulls, kind="invariant"
        )

    @classmethod
    def from_levels(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
        drop_nulls: bool = False,
    ) -> ScenarioPanel:
        return cls._from_tagged_values(
            df=df, prob=prob, drop_nulls=drop_nulls, kind="level"
        )

    @classmethod
    def _from_tagged_values(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None,
        drop_nulls: bool,
        kind: ScenarioPanelKind,
    ) -> ScenarioPanel:
        dates, values = _split_dates_and_values(df)

        if prob is None:
            prob = uniform_probs(values.height)

        if drop_nulls:
            values, dates, prob = cls._drop_null_rows(values, dates, prob)

        return cls(values=values, dates=dates, prob=prob, kind=kind)

    def to_frame(self) -> DataFrame:
        return DataFrame({"date": self.dates}).hstack(self.values)

    def drop_nulls(self) -> ScenarioPanel:
        values, dates, prob = self._drop_null_rows(self.values, self.dates, self.prob)
        if values is self.values:
            return self

        return ScenarioPanel(values=values, dates=dates, prob=prob, kind=self.kind)

    @staticmethod
    def _drop_null_rows(
        values: pl.DataFrame,
        dates: pl.Series,
        prob: ProbVector,
    ) -> tuple[pl.DataFrame, pl.Series, ProbVector]:
        if values.height != prob.shape[0]:
            raise ValueError(f"prob length {prob.shape[0]} != rows {values.height}")
        if len(dates) != values.height:
            raise ValueError(f"dates length {len(dates)} != rows {values.height}")

        null_mask = values.select(pl.any_horizontal(pl.all().is_null())).to_series()

        if not null_mask.any():
            return values, dates, prob

        keep_mask = ~null_mask
        clean = values.filter(keep_mask)
        new_dates = dates.filter(keep_mask)

        dropped_idx = np.flatnonzero(null_mask.to_numpy())
        new_prob = redistribute_prob_mass(prob, dropped_idx)

        return clean, new_dates, new_prob

    def diff(self, lag: int = 1) -> ScenarioPanel:
        if lag < 1:
            raise ValueError("lag must be >= 1")

        diffed = self.values.with_columns(
            [pl.col(c).diff(lag).alias(c) for c in self.values.columns]
        ).slice(lag)
        new_dates = self.dates.slice(lag)
        new_prob = compensate_prob(self.prob, lag)
        new_kind: ScenarioPanelKind = (
            "return" if self.kind == "log_price" else self.kind
        )
        return ScenarioPanel(
            values=diffed, dates=new_dates, prob=new_prob, kind=new_kind
        )

    def with_prob(self, prob: ProbVector) -> ScenarioPanel:
        return ScenarioPanel(
            values=self.values, dates=self.dates, prob=prob, kind=self.kind
        )

    @property
    def asset_names(self) -> list[str]:
        return self.values.columns

    def __len__(self) -> int:
        return self.values.height


def _split_dates_and_values(df: pl.DataFrame) -> tuple[pl.Series, pl.DataFrame]:
    if "date" not in df.columns:
        raise ValueError("ScenarioPanel input must contain a 'date' column")
    return df.get_column("date"), df.drop("date")


def normalize_datetime_series(dates: pl.Series) -> DatetimeSeries:
    if dates.dtype == pl.Date:
        normalized = dates.cast(pl.Datetime)
    elif isinstance(dates.dtype, pl.Datetime):
        normalized = dates
    else:
        raise TypeError(
            f"ScenarioPanel.dates must have Date/Datetime dtype, got {dates.dtype}"
        )
    return normalized.rename("date")
