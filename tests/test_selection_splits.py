from __future__ import annotations

from datetime import datetime

import pytest

from qraft.backtest.selection.splits import walk_forward

DATES = [datetime(2024, 1, 1 + i) for i in range(10)]


def test_walk_forward_rolling() -> None:
    folds = walk_forward(DATES, train_size=3, test_size=2)
    assert [(f.train, f.test) for f in folds] == [
        ((DATES[0], DATES[2]), (DATES[3], DATES[4])),
        ((DATES[2], DATES[4]), (DATES[5], DATES[6])),
        ((DATES[4], DATES[6]), (DATES[7], DATES[8])),
    ]


def test_walk_forward_anchored_expands_train() -> None:
    folds = walk_forward(DATES, train_size=3, test_size=2, anchored=True)
    assert [f.train[0] for f in folds] == [DATES[0], DATES[0], DATES[0]]
    assert [f.train[1] for f in folds] == [DATES[2], DATES[4], DATES[6]]


def test_walk_forward_embargo_shifts_test() -> None:
    folds = walk_forward(DATES, train_size=3, test_size=2, embargo=1)
    # train [0,2], embargo skips d3, test [4,5]
    assert folds[0].train == (DATES[0], DATES[2])
    assert folds[0].test == (DATES[4], DATES[5])


def test_walk_forward_custom_step_overlaps() -> None:
    folds = walk_forward(DATES, train_size=3, test_size=2, step=1)
    assert folds[0].test == (DATES[3], DATES[4])
    assert folds[1].test == (DATES[4], DATES[5])  # step=1 -> overlapping tests


def test_walk_forward_too_short_returns_empty() -> None:
    assert walk_forward(DATES[:3], train_size=3, test_size=2) == []


def test_walk_forward_validates() -> None:
    with pytest.raises(ValueError):
        walk_forward(DATES, train_size=0, test_size=2)
