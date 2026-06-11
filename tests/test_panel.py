import numpy as np
import polars as pl
import pytest

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.distributions import uniform_probs


def test_scenario_panel_rejects_non_finite_values() -> None:
    values = pl.DataFrame({"asset": [0.0, np.nan]})

    with pytest.raises(ValueError, match="finite"):
        ScenarioPanel(values=values, dates=None, prob=uniform_probs(values.height))


def test_from_prices_logs_positive_prices_and_keeps_dates() -> None:
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "asset": [1.0, np.e],
        }
    )

    panel = ScenarioPanel.from_prices(df)

    assert panel.dates is not None
    assert panel.dates.to_list() == ["2024-01-01", "2024-01-02"]
    np.testing.assert_allclose(panel.values.get_column("asset").to_numpy(), [0.0, 1.0])
    np.testing.assert_allclose(panel.prob, [0.5, 0.5])


def test_from_prices_rejects_non_finite_prices() -> None:
    df = pl.DataFrame({"asset": [1.0, np.inf]})

    with pytest.raises(ValueError, match="values contain NaN/inf"):
        ScenarioPanel.from_prices(df)


def test_from_prices_rejects_non_positive_prices() -> None:
    df = pl.DataFrame({"asset": [1.0, 0.0]})

    with pytest.raises(ValueError, match="strictly positive"):
        ScenarioPanel.from_prices(df)


def test_from_log_prices_can_drop_nulls_before_validation() -> None:
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_a": [1.0, None, 3.0],
            "asset_b": [4.0, 5.0, 6.0],
        }
    )
    prob = np.array([0.2, 0.3, 0.5])

    panel = ScenarioPanel.from_log_prices(df, prob=prob, drop_nulls=True)

    assert panel.dates is not None
    assert panel.dates.to_list() == ["2024-01-01", "2024-01-03"]
    assert panel.values.height == 2
    np.testing.assert_allclose(panel.prob, [2.0 / 7.0, 5.0 / 7.0])
