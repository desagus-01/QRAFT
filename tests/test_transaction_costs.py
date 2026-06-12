import numpy as np
import pytest

from qraft.construction.optimization.moments import HorizonMoments
from qraft.construction.optimization.objectives.handlers import (
    HoldingCostHandler,
    TransactionCostHandler,
)
from qraft.construction.optimization.objectives.specs import (
    HoldingCost,
    TransactionCost,
)
from qraft.construction.optimization.presets import (
    _default_holding_cost,
    _default_transaction_cost,
)
from qraft.construction.policies import MPOPolicy
from qraft.forecast.forecast_paths import ForecastPaths


def _moments() -> HorizonMoments:
    return HorizonMoments(
        assets=["A", "B"],
        correlations=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        covariances=np.array([[[0.04, 0.0], [0.0, 0.09]]]),
        mean=np.array([[0.01, 0.02]]),
        scenario_returns=np.array([[[0.01, 0.02]], [[0.02, 0.01]]]),
        scenario_probs=np.array([0.5, 0.5]),
        cash_return=np.array([0.0]),
    )


def _params(spec: TransactionCost) -> dict:
    return TransactionCostHandler().allocate(spec, horizons=1, n_assets=2)


def test_pershare_cost_requires_prices() -> None:
    spec = TransactionCost(pershare_cost=0.005, market_impact=0.0)

    with pytest.raises(ValueError, match="prices"):
        TransactionCostHandler().update(
            spec,
            _params(spec),
            {"moments": _moments()},
        )


def test_market_impact_requires_weight_space_volume() -> None:
    spec = TransactionCost(pershare_cost=0.0, market_impact=0.3)

    with pytest.raises(ValueError, match="volume"):
        TransactionCostHandler().update(
            spec,
            _params(spec),
            {"moments": _moments()},
        )


def test_transaction_cost_coefficients_use_required_inputs() -> None:
    spec = TransactionCost(
        cost=0.0005,
        pershare_cost=0.005,
        market_impact=0.3,
        exponent=1.5,
    )
    params = _params(spec)

    TransactionCostHandler().update(
        spec,
        params,
        {
            "moments": _moments(),
            "prices": np.array([10.0, 20.0]),
            "volume": np.array([2.0, 8.0]),
        },
    )

    np.testing.assert_allclose(params["tc_linear"].value, [0.001, 0.00075])
    np.testing.assert_allclose(
        params["tc_impact"].value,
        [
            0.3 * 0.2 / np.sqrt(2.0),
            0.3 * 0.3 / np.sqrt(8.0),
        ],
    )


def test_preset_cost_defaults_are_neutral_and_percent_scaled() -> None:
    transaction_cost = _default_transaction_cost()

    assert transaction_cost.market_impact == 0.0
    assert transaction_cost.c_bias == 0.0
    assert _default_holding_cost().long_fees == 0.119


def test_policy_prices_are_taken_from_forecasts_in_moment_asset_order() -> None:
    forecasts = ForecastPaths(
        asset_paths={
            "A": np.ones((2, 1)),
            "B": np.ones((2, 1)),
        },
        path_probs=np.array([0.5, 0.5]),
        initial_prices={
            "B": 20.0,
            "A": 10.0,
        },
    )

    np.testing.assert_allclose(
        MPOPolicy._initial_prices_for_assets(forecasts, ["A", "B"]),
        [10.0, 20.0],
    )


def test_holding_cost_dividends_are_decimal_rates_like_fees() -> None:
    spec = HoldingCost(short_fees=0.01, long_fees=0.02, dividends=0.03)
    params = HoldingCostHandler().allocate(spec, horizons=1, n_assets=2)

    HoldingCostHandler().update(spec, params, {})

    assert params["short_rate"].value == spec.short_fees / spec.periods_per_year
    assert params["long_rate"].value == spec.long_fees / spec.periods_per_year
    assert params["div_rate"].value == spec.dividends / spec.periods_per_year
