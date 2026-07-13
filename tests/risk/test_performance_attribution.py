from datetime import datetime

import numpy as np
import pytest

from qraft.risk.performance_attribution import portfolio_factor_attribution


def test_portfolio_factor_attribution_recovers_exposures_and_joint_panel() -> None:
    factor_a_prices = np.array([[101.0], [102.0], [99.0], [98.0], [103.0], [97.0]])
    factor_b_prices = np.array([[52.0], [49.0], [51.0], [48.0], [53.0], [47.0]])
    factor_a = factor_a_prices[:, 0] / 100.0 - 1.0
    factor_b = factor_b_prices[:, 0] / 50.0 - 1.0
    portfolio = 0.01 + 2.0 * factor_a - 0.5 * factor_b

    attribution = portfolio_factor_attribution(
        portfolio,
        np.full(6, 1 / 6),
        {"A": factor_a_prices, "B": factor_b_prices},
        {"A": 100.0, "B": 50.0},
        horizon=0,
        date=datetime(2024, 1, 1),
    )

    assert attribution.exposures["A"] == pytest.approx(2.0)
    assert attribution.exposures["B"] == pytest.approx(-0.5)
    assert attribution.r2 == pytest.approx(1.0)
    assert attribution.full_exposures["idiosyncratic"] == 1.0
    assert set(attribution.joint_distribution.columns) == {
        "A",
        "B",
        "idiosyncratic",
        "portfolio_performance",
    }


def test_portfolio_factor_attribution_auto_selection_filters_factors() -> None:
    factor_a_prices = np.array(
        [[101.0], [102.0], [99.0], [98.0], [103.0], [97.0], [104.0], [96.0]]
    )
    factor_b_prices = np.full((8, 1), 50.0)
    factor_a = factor_a_prices[:, 0] / 100.0 - 1.0
    portfolio = 3.0 * factor_a

    attribution = portfolio_factor_attribution(
        portfolio,
        np.full(8, 1 / 8),
        {"A": factor_a_prices, "B": factor_b_prices},
        {"A": 100.0, "B": 50.0},
        horizon=0,
        date=datetime(2024, 1, 1),
        eq_type="nc",
        auto_select_factors=True,
        criterion="r2",
    )

    assert attribution.factor_names == ["A"]
    assert attribution.exposures["A"] == pytest.approx(3.0)
