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
from qraft.core.scenarios.view_types import (
    CorrView,
    EntropyPoolingResult,
    MeanView,
    RankingView,
    StdView,
)


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


def test_entropy_pooling_confidence_diagnostics_describe_blended_posterior() -> None:
    full = entropy_pooling_probs(_panel(), [MeanView("A", ">=", 1.5)])
    blended = entropy_pooling_probs(
        _panel(), [MeanView("A", ">=", 1.5)], confidence=0.5
    )

    expected_ens = np.exp(-np.sum(blended.posterior * np.log(blended.posterior)))

    assert blended.diagnostics.ens_posterior == pytest.approx(expected_ens)
    assert blended.diagnostics.ens_posterior != pytest.approx(
        full.diagnostics.ens_posterior
    )
    assert not blended.diagnostics.constraints[0]["active"]


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


def test_ranking_view_requires_at_least_two_distinct_assets() -> None:
    with pytest.raises(
        ValueError,
        match="RankingView needs at least two assets to express an ordering; got 1.",
    ):
        RankingView(["A"])

    with pytest.raises(ValueError, match="duplicate assets"):
        RankingView(["A", "A"])


def test_std_view_pins_posterior_mean_to_anchor() -> None:
    result = entropy_pooling_probs(
        _panel(), [MeanView("A", "==", 1.0), StdView("A", ">=", 0.75)]
    )

    values = _panel().values["A"].to_numpy()
    posterior_mean = float(values @ result.posterior)

    assert posterior_mean == pytest.approx(1.0, abs=1e-5)
    assert len(result.diagnostics.constraints) == 3
    assert result.diagnostics.constraints[2]["risk_driver"] == ("A", "anchor:mean")


def test_inequality_mean_cannot_anchor_std_or_corr_view() -> None:
    with pytest.raises(
        ValueError,
        match="StdView on 'A' cannot be anchored to an inequality mean view",
    ):
        entropy_pooling_probs(
            _panel(), [MeanView("A", ">=", 1.0), StdView("A", ">=", 0.5)]
        )

    panel = ScenarioPanel.from_levels(
        pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "A": [0.0, 1.0, 2.0],
                "B": [2.0, 1.0, 0.0],
            }
        )
    )
    with pytest.raises(
        ValueError,
        match="StdView on 'A' cannot be anchored to an inequality mean view",
    ):
        entropy_pooling_probs(
            panel, [MeanView("A", ">=", 1.0), CorrView(("A", "B"), "<=", 0.0)]
        )
