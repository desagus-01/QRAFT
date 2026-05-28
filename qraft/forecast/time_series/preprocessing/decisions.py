import logging
from typing import Literal

import numpy as np

from time_series.preprocessing.types import (
    TransformDecision,
)
from time_series.selection.trend import AssetTrendDiagnostic, TrendSelection
from time_series.transforms.detrend import TrendCandidate
from time_series.tests.seasonality import SEASONAL_MAP, SeasonalityPeriodTest

logger = logging.getLogger(__name__)


def _accepted_candidate(selection: TrendSelection | None) -> TrendCandidate | None:
    """Return selected candidate only if it satisfies the selection's own threshold policy."""
    if selection is None or not selection.transformation_needed:
        return None
    return selection.selected


def detrend_decision_rule(
    detrend_res: dict[str, AssetTrendDiagnostic],
    assets: list[str],
    tie_break: Literal["polynomial", "difference"] = "difference",
) -> dict[str, TransformDecision]:
    """Choose the lowest-order winning trend transform per asset.

    Strategy:
        Prefer the smallest order tie-break
    """
    trend_trans = {}

    for asset in assets:
        deterministic = detrend_res[asset].deterministic
        stochastic = detrend_res[asset].stochastic
        deterministic_res = _accepted_candidate(deterministic)
        stochastic_res = _accepted_candidate(stochastic)
        candidates = []

        if deterministic_res is not None:
            candidates.append(("polynomial", deterministic_res.order))

        if stochastic_res is not None:
            candidates.append(("difference", stochastic_res.order))

        if not candidates:
            continue

        transformation, order = min(candidates, key=lambda x: (x[1], x[0] != tie_break))

        trend_trans[asset] = TransformDecision(kind=transformation, order=order)

    return trend_trans


def _expand_period_to_harmonics(period_label: str) -> list[tuple[str, float]]:
    """
    Given a period label ('weekly', 'monthly', ...), return a list of
    (label_for_harmonic, omega_radians) covering the full harmonic set.
    The label is augmented (e.g., 'monthly_h2'); _deseason_apply ignores labels anyway.
    """
    P = SEASONAL_MAP[period_label]
    base = 2 * np.pi / P

    out: list[tuple[str, float]] = []
    H = (P - 1) // 2
    for h in range(1, H + 1):
        out.append((f"{period_label}_h{h}", h * base))

    if P % 2 == 0:
        out.append((f"{period_label}_nyq", np.pi))  # Nyquist cosine

    return out


def deseason_decision_rule(
    seasonality_diagnostic: dict[str, list[SeasonalityPeriodTest]],
) -> dict[str, list[tuple[str, float]]]:
    """
    Extract significant seasonal periods and expand each to its harmonic set.
    """
    decision: dict[str, list[tuple[str, float]]] = {}

    for asset, tests in seasonality_diagnostic.items():
        freqs: list[tuple[str, float]] = []
        for t in tests:
            if t.evidence_of_seasonality:
                freqs.extend(_expand_period_to_harmonics(t.seasonal_period))

        #  deduplicate angular frequencies if multiple periods collide
        if freqs:
            freqs_sorted = sorted(freqs, key=lambda kv: kv[1])
            dedup: list[tuple[str, float]] = []
            for lab, w in freqs_sorted:
                if not dedup or not np.isclose(w, dedup[-1][1], rtol=0, atol=1e-12):
                    dedup.append((lab, w))
            decision[asset] = dedup
        else:
            decision[asset] = []

    return decision
