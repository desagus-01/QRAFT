from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward split, as inclusive ``(start, end)`` rebalance dates."""

    train: tuple[datetime, datetime]
    test: tuple[datetime, datetime]


def walk_forward(
    dates: Sequence[datetime],
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    anchored: bool = False,
) -> list[Fold]:
    """Walk-forward folds over an ordered list of rebalance ``dates``."""
    if train_size < 1 or test_size < 1 or embargo < 0:
        raise ValueError("train_size, test_size must be >= 1 and embargo >= 0")
    advance = test_size if step is None else step
    if advance < 1:
        raise ValueError("step must be >= 1")

    ordered = list(dates)
    n = len(ordered)
    folds: list[Fold] = []
    start = 0
    while True:
        train_hi = start + train_size - 1
        test_lo = train_hi + 1 + embargo
        test_hi = test_lo + test_size - 1
        if test_hi >= n:
            break
        train_lo = 0 if anchored else start
        folds.append(
            Fold(
                train=(ordered[train_lo], ordered[train_hi]),
                test=(ordered[test_lo], ordered[test_hi]),
            )
        )
        start += advance
    return folds
