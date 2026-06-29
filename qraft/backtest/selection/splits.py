from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations

import numpy as np

DateRange = tuple[datetime, datetime]


@dataclass(frozen=True, slots=True)
class Fold:
    """One split with inclusive train and test date ranges."""

    train: DateRange
    test: DateRange


@dataclass(frozen=True, slots=True)
class CombinatorialFold:
    """One CPCV split with train/test unions and held-out test group ids."""

    train: tuple[DateRange, ...]
    test: tuple[DateRange, ...]
    test_groups: tuple[int, ...]


def ordered_unique_dates(dates: Sequence[datetime]) -> list[datetime]:
    """Return sorted unique dates and reject empty inputs early."""
    ordered = sorted(set(dates))
    if not ordered:
        raise ValueError("dates must contain at least one date")
    return ordered


def walk_forward_folds(
    dates: Sequence[datetime],
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    anchored: bool = False,
) -> list[Fold]:
    """Walk-forward folds over rebalance dates."""
    advance = _validate_walk_forward_params(train_size, test_size, step, embargo)
    ordered = ordered_unique_dates(dates)
    n_dates = len(ordered)
    folds: list[Fold] = []
    start_index = 0
    while True:
        train_end_index = start_index + train_size - 1
        test_start_index = train_end_index + 1 + embargo
        test_end_index = test_start_index + test_size - 1
        if test_end_index >= n_dates:
            break
        train_start_index = 0 if anchored else start_index
        folds.append(
            Fold(
                train=(ordered[train_start_index], ordered[train_end_index]),
                test=(ordered[test_start_index], ordered[test_end_index]),
            )
        )
        start_index += advance
    return folds


def combinatorial_purged_folds(
    dates: Sequence[datetime],
    *,
    n_groups: int,
    n_test_groups: int = 2,
    purge: int = 0,
    embargo: int = 0,
) -> list[CombinatorialFold]:
    """All C(n_groups, n_test_groups) purged/embargoed CPCV folds."""
    _validate_combinatorial_params(n_groups, n_test_groups, purge, embargo)
    ordered = ordered_unique_dates(dates)
    n_dates = len(ordered)
    if n_dates < n_groups:
        raise ValueError(f"need >= n_groups={n_groups} dates, got {n_dates}")
    group_bounds = np.linspace(0, n_dates, n_groups + 1).astype(int)
    if any(group_bounds[g + 1] - group_bounds[g] < 1 for g in range(n_groups)):
        raise ValueError(
            f"too few dates ({n_dates}) to form {n_groups} non-empty groups"
        )

    folds: list[CombinatorialFold] = []
    for test_group_ids in combinations(range(n_groups), n_test_groups):
        test_ranges = tuple(
            (ordered[group_bounds[group_id]], ordered[group_bounds[group_id + 1] - 1])
            for group_id in test_group_ids
        )
        blocked_indices = _blocked_indices(
            test_group_ids, group_bounds, n_dates, purge, embargo
        )
        train_indices = [
            date_index
            for group_id in range(n_groups)
            if group_id not in test_group_ids
            for date_index in range(
                int(group_bounds[group_id]), int(group_bounds[group_id + 1])
            )
            if date_index not in blocked_indices
        ]
        folds.append(
            CombinatorialFold(
                train=runs_to_ranges(train_indices, ordered),
                test=test_ranges,
                test_groups=tuple(test_group_ids),
            )
        )
    return folds


def runs_to_ranges(
    indices: Sequence[int], ordered_dates: Sequence[datetime]
) -> tuple[DateRange, ...]:
    """Collapse sorted date indices into contiguous inclusive date ranges."""
    if not indices:
        return ()
    sorted_indices = sorted(indices)
    ranges: list[DateRange] = []
    range_start_index = previous_index = sorted_indices[0]
    for date_index in sorted_indices[1:]:
        if date_index == previous_index + 1:
            previous_index = date_index
        else:
            ranges.append(
                (ordered_dates[range_start_index], ordered_dates[previous_index])
            )
            range_start_index = previous_index = date_index
    ranges.append((ordered_dates[range_start_index], ordered_dates[previous_index]))
    return tuple(ranges)


def _validate_walk_forward_params(
    train_size: int,
    test_size: int,
    step: int | None,
    embargo: int,
) -> int:
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    advance = test_size if step is None else step
    if advance < 1:
        raise ValueError("step must be >= 1")
    return advance


def _validate_combinatorial_params(
    n_groups: int,
    n_test_groups: int,
    purge: int,
    embargo: int,
) -> None:
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups)")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0")


def _blocked_indices(
    test_group_ids: Sequence[int],
    group_bounds: np.ndarray,
    n_dates: int,
    purge: int,
    embargo: int,
) -> set[int]:
    blocked_indices: set[int] = set()
    for group_id in test_group_ids:
        group_start_index = int(group_bounds[group_id])
        group_end_index = int(group_bounds[group_id + 1] - 1)
        blocked_indices.update(
            range(
                max(0, group_start_index - purge),
                min(n_dates, group_end_index + embargo + 1),
            )
        )
    return blocked_indices
