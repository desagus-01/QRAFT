from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Sequence


def infer_periods_per_year(dates: Sequence[datetime]) -> float:
    if len(dates) < 2:
        raise ValueError("At least two dates are required to infer periods_per_year.")
    deltas = [
        (curr - prev).total_seconds() / 86_400.0 for prev, curr in zip(dates, dates[1:])
    ]
    if any(delta <= 0 for delta in deltas):
        raise ValueError("dates must be strictly increasing to infer periods_per_year.")
    median_delta = median(deltas)
    if median_delta == 1.0:
        return 252.0
    return 365.25 / median_delta


def resolve_periods_per_year(
    dates: Sequence[datetime], configured: float | None, *, rtol: float = 0.2
) -> float:
    if len(dates) < 2 and configured is not None:
        if configured <= 0:
            raise ValueError("periods_per_year must be > 0")
        return configured
    inferred = infer_periods_per_year(dates)
    if configured is None:
        return inferred
    if configured <= 0:
        raise ValueError("periods_per_year must be > 0")
    if abs(configured - inferred) > rtol * inferred:
        raise ValueError(
            "periods_per_year is inconsistent with observed cadence: "
            f"configured={configured:.6g}, inferred={inferred:.6g}."
        )
    return configured
