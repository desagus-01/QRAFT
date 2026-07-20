from datetime import datetime, timedelta

import numpy as np
import polars as pl
from scipy.optimize import brentq

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.entropy_pooling import entropy_pooling
from qraft.core.scenarios.view_types import MeanView


def test_mean_equality_view_matches_exponential_tilting() -> None:
    values = np.array([-0.04, -0.01, 0.0, 0.02, 0.06])
    target = 0.015
    panel = ScenarioPanel.from_levels(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)],
                "A": values,
            }
        )
    )

    result = entropy_pooling(panel, [MeanView("A", "==", target)])
    prior = np.asarray(panel.prob)

    def tilted_mean(lagrange: float) -> float:
        logits = np.log(prior) + lagrange * values
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return float(weights @ values)

    lagrange = brentq(lambda value: tilted_mean(value) - target, -1_000, 1_000)
    logits = np.log(prior) + lagrange * values
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()

    np.testing.assert_allclose(result.posterior, expected, atol=1e-7, rtol=1e-7)
