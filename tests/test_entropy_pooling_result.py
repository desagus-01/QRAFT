import logging
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.entropy_pooling import (
    InfeasibleViewsError,
    entropy_pooling_probs,
)
from qraft.core.scenarios.view_types import MeanView
from qraft.core.scenarios.viewed import EntropyPoolingResult


def _panel() -> ScenarioPanel:
    return ScenarioPanel.from_levels(
        pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "A": [0.0, 1.0, 2.0],
            }
        )
    )


def test_entropy_pooling_probs_returns_result_with_diagnostics() -> None:
    result = entropy_pooling_probs(_panel(), [MeanView("A", ">=", 1.5)])

    assert isinstance(result, EntropyPoolingResult)
    np.testing.assert_allclose(result.posterior.sum(), 1.0)
    assert result.diagnostics.ens_prior == pytest.approx(3.0)
    assert result.diagnostics.ens_posterior > 0.0
    assert result.diagnostics.solver_status in ("optimal", "optimal_inaccurate")
    assert not result.diagnostics.ens_collapsed
    assert len(result.diagnostics.constraints) == 1
    assert result.diagnostics.constraints[0]["risk_driver"] == "A"


def test_entropy_pooling_confidence_blends_posterior_before_packaging() -> None:
    full = entropy_pooling_probs(_panel(), [MeanView("A", ">=", 1.5)])
    blended = entropy_pooling_probs(
        _panel(), [MeanView("A", ">=", 1.5)], confidence=0.5
    )

    expected = 0.5 * full.posterior + 0.5 * _panel().prob
    np.testing.assert_allclose(blended.posterior, expected)


def test_entropy_pooling_infeasible_views_error_names_constraints() -> None:
    with pytest.raises(InfeasibleViewsError, match="A >= 3.0"):
        entropy_pooling_probs(_panel(), [MeanView("A", ">=", 3.0)])


def test_entropy_pooling_warns_when_ens_collapses(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="qraft.core.probability.entropy_pooling")

    result = entropy_pooling_probs(
        _panel(), [MeanView("A", ">=", 1.5)], ens_warn_ratio=1.0
    )

    assert result.diagnostics.ens_collapsed
    assert any(record.qraft_event == "views.ens_collapsed" for record in caplog.records)
