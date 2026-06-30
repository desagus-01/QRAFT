from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.distributions import uniform_probs


def test_scenario_panel_rejects_non_finite_values() -> None:
    values = pl.DataFrame({"asset": [0.0, np.nan]})
    dates = pl.Series("date", [datetime(2024, 1, 1), datetime(2024, 1, 2)])

    with pytest.raises(ValueError, match="finite"):
        ScenarioPanel(values=values, dates=dates, prob=uniform_probs(values.height))


def test_from_prices_logs_positive_prices_and_keeps_dates() -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "asset": [1.0, np.e],
        }
    )

    panel = ScenarioPanel.from_prices(df)

    assert panel.dates.to_list() == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    assert panel.kind == "log_price"
    np.testing.assert_allclose(panel.values.get_column("asset").to_numpy(), [0.0, 1.0])
    np.testing.assert_allclose(panel.prob, [0.5, 0.5])


def test_from_prices_requires_date_column() -> None:
    with pytest.raises(ValueError, match="'date' column"):
        ScenarioPanel.from_prices(pl.DataFrame({"asset": [1.0, 2.0]}))


def test_scenario_panel_rejects_non_temporal_dates() -> None:
    with pytest.raises(TypeError, match="Date/Datetime"):
        ScenarioPanel.from_prices(
            pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "asset": [1.0, 2.0]})
        )


def test_from_prices_rejects_non_finite_prices() -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "asset": [1.0, np.inf],
        }
    )

    with pytest.raises(ValueError, match="values contain NaN/inf"):
        ScenarioPanel.from_prices(df)


def test_from_prices_rejects_non_positive_prices() -> None:
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "asset": [1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="strictly positive"):
        ScenarioPanel.from_prices(df)


def test_from_log_prices_can_drop_nulls_before_validation() -> None:
    df = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "asset_a": [1.0, None, 3.0],
            "asset_b": [4.0, 5.0, 6.0],
        }
    )
    prob = np.array([0.2, 0.3, 0.5])

    panel = ScenarioPanel.from_log_prices(df, prob=prob, drop_nulls=True)

    assert panel.dates.to_list() == [datetime(2024, 1, 1), datetime(2024, 1, 3)]
    assert panel.kind == "log_price"
    assert panel.values.height == 2
    np.testing.assert_allclose(panel.prob, [2.0 / 7.0, 5.0 / 7.0])


def test_direct_scenario_panel_defaults_to_level_kind() -> None:
    panel = ScenarioPanel(
        values=pl.DataFrame({"asset": [1.0, 2.0]}),
        dates=pl.Series("date", [datetime(2024, 1, 1), datetime(2024, 1, 2)]),
        prob=uniform_probs(2),
    )

    assert panel.kind == "level"


def test_scenario_panel_copies_and_freezes_prob() -> None:
    prob = np.array([0.5, 0.5])
    panel = ScenarioPanel(
        values=pl.DataFrame({"asset": [1.0, 2.0]}),
        dates=pl.Series("date", [datetime(2024, 1, 1), datetime(2024, 1, 2)]),
        prob=prob,
    )

    prob[0] = 1.0

    np.testing.assert_allclose(panel.prob, [0.5, 0.5])
    assert not panel.prob.flags.writeable


def test_scenario_panel_rejects_invalid_prob() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ScenarioPanel(
            values=pl.DataFrame({"asset": [1.0, 2.0]}),
            dates=pl.Series("date", [datetime(2024, 1, 1), datetime(2024, 1, 2)]),
            prob=np.array([0.5, 0.4]),
        )


def test_log_price_diff_is_tagged_as_return() -> None:
    panel = ScenarioPanel.from_log_prices(
        pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "asset": [0.0, 0.1, 0.3],
            }
        )
    )

    diffed = panel.diff()

    assert diffed.kind == "return"
