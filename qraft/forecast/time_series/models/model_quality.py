from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy._typing import NDArray

from policy import QualityConfig

QualityEvent = Literal[
    "MEAN_FALLBACK_DEMEAN",
    "MEAN_FALLBACK_BEST_IC_NO_DIAG_PASS",
    "VOL_FALLBACK_BEST_IC_NO_DIAG_PASS",
]

Severity = Literal["low", "medium", "high"]

EVENT_SEVERITY: dict[QualityEvent, Severity] = {
    "MEAN_FALLBACK_DEMEAN": "medium",
    "MEAN_FALLBACK_BEST_IC_NO_DIAG_PASS": "high",
    "VOL_FALLBACK_BEST_IC_NO_DIAG_PASS": "high",
}

# QualityConfig is defined in globals and re-exported here for convenience.
__all__ = [
    "QualityConfig",
    "SelectionAudit",
    "ModelQuality",
    "score_audit",
    "crps_ensemble",
]


@dataclass(slots=True)
class SelectionAudit:
    events: list[QualityEvent]
    notes: list[str]

    def add_event(self, event: QualityEvent, note: str | None = None) -> None:
        self.events.append(event)
        if note:
            self.notes.append(note)


@dataclass(frozen=True, slots=True)
class ModelQuality:
    score: float
    grade: Literal["A", "B", "C", "D"]
    reason_codes: tuple[QualityEvent, ...]


def _grade(
    score: float, cfg: QualityConfig | None = None
) -> Literal["A", "B", "C", "D"]:
    if cfg is None:
        cfg = QualityConfig()
    if score >= cfg.grade_a_threshold:
        return "A"
    if score >= cfg.grade_b_threshold:
        return "B"
    if score >= cfg.grade_c_threshold:
        return "C"
    return "D"


def score_audit(audit: SelectionAudit, cfg: QualityConfig) -> ModelQuality:
    score = 100.0

    for event in audit.events:
        severity = EVENT_SEVERITY.get(event, "low")
        if severity == "high":
            score -= cfg.penalty_high
        elif severity == "medium":
            score -= cfg.penalty_medium
        else:
            score -= cfg.penalty_low

    score = max(0.0, min(100.0, score))

    return ModelQuality(
        score=round(score, 2),
        grade=_grade(score, cfg),
        reason_codes=tuple(audit.events),
    )


def crps_ensemble(
    forecast: NDArray[np.floating],
    obs: NDArray[np.floating],
    method: str = "fair",
    axis: int = 0,
) -> NDArray[np.floating]:
    if method not in {"fair", "ecdf"}:
        raise ValueError("method must be 'fair' or 'ecdf'")
    if axis != 0:
        forecast = np.moveaxis(forecast, axis, 0)
    m = forecast.shape[0]
    if m < 2 and method == "fair":
        raise ValueError("Need at least 2 ensemble members for method='fair'")
    fcst_obs_term = np.mean(np.abs(forecast - obs), axis=0)
    fcst_sorted = np.sort(forecast, axis=0)
    i = np.arange(m).reshape((m,) + (1,) * (forecast.ndim - 1))
    coeffs = 2 * i - m + 1
    spread_numerator = 2.0 * np.sum(fcst_sorted * coeffs, axis=0)
    if method == "ecdf":
        spread_term = spread_numerator / (2.0 * m * m)
    else:
        spread_term = spread_numerator / (2.0 * m * (m - 1))
    return fcst_obs_term - spread_term
