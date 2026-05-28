from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame

from probability.distributions import uniform_probs
from scenarios.types import ProbVector, validate_prob_vector


def redistribute_prob_mass(
    prob: ProbVector,
    dropped_idx: NDArray[np.int_],
) -> ProbVector:
    """Drop entries at ``dropped_idx`` and spread their mass across the rest."""
    if dropped_idx.size == 0:
        return prob.copy()

    kept = np.delete(prob, dropped_idx)
    if kept.size == 0:
        raise ValueError("Cannot redistribute: all entries would be removed")

    mass_removed = float(prob[dropped_idx].sum())
    return kept + mass_removed / kept.size


def compensate_prob(prob: ProbVector, n_remove: int) -> ProbVector:
    """Drop the first ``n_remove`` entries and redistribute their mass.

    Convenience wrapper around :func:`redistribute_prob_mass` for the common
    "leading rows were trimmed" case (e.g. after differencing or lag alignment).
    """
    if n_remove < 0:
        raise ValueError("n_remove must be non-negative")
    return redistribute_prob_mass(prob, np.arange(n_remove, dtype=np.int_))


@dataclass(frozen=True)
class ScenarioPanel:
    """Canonical (values × dates × prob) triple used throughout the pipeline.

    Invariants (enforced in ``__post_init__``):
      * ``values`` has no ``date`` column; dates live in ``dates``.
      * ``values.height == len(prob)`` and ``== len(dates)`` when dates present.
      * ``prob`` is a valid :data:`ProbVector`: 1-D, finite, non-negative,
        summing to 1. Stored as a read-only array.
      * ``dates.name == 'date'`` when present.
      * The panel is non-empty.
    """

    values: pl.DataFrame
    dates: pl.Series | None  # None so that simulations can also use this
    prob: ProbVector

    def __post_init__(self) -> None:
        if "date" in self.values.columns:
            raise ValueError(
                "AssetPanel.values must not contain a 'date' column; "
                "use AssetPanel.from_frame() to separate it."
            )
        if self.values.height == 0:
            raise ValueError("AssetPanel cannot be empty")

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
    def from_frame(
        cls,
        df: pl.DataFrame,
        prob: ProbVector | None = None,
    ) -> ScenarioPanel:
        """Build a panel from a DataFrame, splitting out the ``date`` column.

        A uniform prior is assigned when ``prob`` is not provided.
        """
        if "date" in df.columns:
            dates: pl.Series | None = df.get_column("date")
            values = df.drop("date")
        else:
            dates = None
            values = df

        if prob is None:
            prob = uniform_probs(values.height)

        return cls(values=values, dates=dates, prob=prob)

    def to_frame(self) -> DataFrame:
        if self.dates is None:
            return self.values

        return DataFrame({"date": self.dates}).hstack(self.values)

    def drop_nulls(self) -> ScenarioPanel:
        """Drop every row that contains a null in any column.

        The probability mass of each dropped row is redistributed evenly
        across the remaining rows (unlike a simple prefix-trim, this is
        correct for nulls that appear anywhere in the panel).
        """
        null_mask = self.values.select(
            pl.any_horizontal(pl.all().is_null())
        ).to_series()

        if not bool(null_mask.any()):
            return self

        keep_mask = ~null_mask
        clean = self.values.filter(keep_mask)
        new_dates = self.dates.filter(keep_mask) if self.dates is not None else None

        dropped_idx = np.flatnonzero(null_mask.to_numpy())
        new_prob = redistribute_prob_mass(self.prob, dropped_idx)

        return ScenarioPanel(values=clean, dates=new_dates, prob=new_prob)

    def diff(self, lag: int = 1) -> ScenarioPanel:
        """Apply a ``lag``-step first-difference to every column.

        The leading ``lag`` rows are dropped (they become null) and their
        probability mass is redistributed across the remainder.
        """
        if lag < 1:
            raise ValueError("lag must be >= 1")

        diffed = self.values.with_columns(
            [pl.col(c).diff(lag).alias(c) for c in self.values.columns]
        ).slice(lag)
        new_dates = self.dates.slice(lag) if self.dates is not None else None
        new_prob = compensate_prob(self.prob, lag)
        return ScenarioPanel(values=diffed, dates=new_dates, prob=new_prob)

    def with_prob(self, prob: ProbVector) -> ScenarioPanel:
        """Return a new panel with a replaced probability vector."""
        return ScenarioPanel(values=self.values, dates=self.dates, prob=prob)

    def map_values_same_rows(self, values: pl.DataFrame) -> ScenarioPanel:
        """Return a new panel with replaced values; row count must be unchanged.

        ``dates`` and ``prob`` are always preserved.  Raises if ``values`` has
        a different number of rows, making it impossible to silently drop dates.
        Use :meth:`filter_rows` or :meth:`diff` / :meth:`drop_nulls` when the
        row count must change.
        """
        if values.height != self.values.height:
            raise ValueError(
                f"map_values_same_rows requires identical row count; "
                f"got {values.height} vs {self.values.height}. "
                "Use filter_rows() or diff() / drop_nulls() for row-changing ops."
            )
        return ScenarioPanel(values=values, dates=self.dates, prob=self.prob)

    def filter_rows(self, mask: pl.Series) -> ScenarioPanel:
        """Keep rows where *mask* is True, updating ``dates`` and ``prob``.

        Probability mass from dropped rows is redistributed evenly across the
        survivors (same policy as :meth:`drop_nulls`).
        """
        if len(mask) != self.values.height:
            raise ValueError(f"mask length {len(mask)} != rows {self.values.height}")
        if not bool(mask.any()):
            raise ValueError("filter_rows would remove every row")

        new_values = self.values.filter(mask)
        new_dates = self.dates.filter(mask) if self.dates is not None else None
        dropped_idx = np.flatnonzero((~mask).to_numpy())
        new_prob = redistribute_prob_mass(self.prob, dropped_idx)
        return ScenarioPanel(values=new_values, dates=new_dates, prob=new_prob)

    @property
    def has_dates(self) -> bool:
        """True when a row-aligned ``date`` series is present."""
        return self.dates is not None

    def require_dates(self) -> pl.Series:
        """Return the ``date`` series or raise if the panel has none.

        Prefer this over ``panel.dates`` at call sites that genuinely need
        dates; it makes the dependency explicit and the error message clear.
        """
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
