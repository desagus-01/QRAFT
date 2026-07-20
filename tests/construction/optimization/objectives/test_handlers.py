import cvxpy as cp
import numpy as np
import pytest

from qraft.construction.optimization.inputs import OptimizerInputs, psd_sqrt_factor
from qraft.construction.optimization.objectives.handlers import (
    CVaRCuttingPlaneHandler,
    CVaRRiskHandler,
    CashReturnHandler,
    CovarianceRiskHandler,
    ExpectedReturnHandler,
    HoldingCostHandler,
    TransactionCostHandler,
)
from qraft.construction.optimization.objectives.specs import (
    CVaRCuttingPlane,
    CVaRRisk,
    CashReturn,
    CovarianceRisk,
    ExpectedReturn,
    HoldingCost,
    TransactionCost,
    holding_cost_rates,
    transaction_cost_coeffs,
)


def _inputs(**kwargs) -> OptimizerInputs:
    defaults = dict(assets=["A", "B"])
    if not any(
        key in kwargs
        for key in (
            "mean",
            "covariances",
            "correlations",
            "scenario_returns",
            "cash_return",
            "cov_factor",
        )
    ):
        defaults["mean"] = np.array([[0.01, 0.02], [0.03, 0.04]])
    defaults.update(kwargs)
    return OptimizerInputs(**defaults)


def test_expected_return_handler_updates_mean_and_applies_decay() -> None:
    spec = ExpectedReturn(decay=0.5)
    handler = ExpectedReturnHandler()
    params = handler.allocate(spec, horizons=2, n_assets=2)
    optimizer_inputs = _inputs()

    handler.update(spec, params, {"optimizer_inputs": optimizer_inputs})
    weights = cp.Parameter(2, value=np.array([1.0, 2.0]))
    expr, constraints = handler.compile(spec, params, weights, weights, horizon=1)

    assert constraints == []
    assert np.array_equal(params["mean"].value, optimizer_inputs.mean)
    assert expr.value == pytest.approx(0.5 * (0.03 * 1.0 + 0.04 * 2.0))


def test_cash_return_handler_broadcasts_scalar_and_requires_cash_expression() -> None:
    spec = CashReturn()
    handler = CashReturnHandler()
    params = handler.allocate(spec, horizons=2, n_assets=2)

    class Inputs:
        def require_cash_return(self):
            return np.array([0.001])

    handler.update(spec, params, {"optimizer_inputs": Inputs()})

    assert np.array_equal(params["cash_return"].value, np.array([0.001, 0.001]))

    with pytest.raises(ValueError, match="requires cash_h"):
        handler.compile(spec, params, cp.Variable(2), cp.Variable(2), horizon=0)


def test_cash_return_handler_rejects_wrong_vector_length() -> None:
    spec = CashReturn()
    handler = CashReturnHandler()
    params = handler.allocate(spec, horizons=2, n_assets=2)

    class Inputs:
        def require_cash_return(self):
            return np.array([0.001, 0.002, 0.003])

    with pytest.raises(ValueError, match="built with 2 horizon"):
        handler.update(spec, params, {"optimizer_inputs": Inputs()})


def test_covariance_risk_handler_uses_cov_factor_when_available() -> None:
    spec = CovarianceRisk()
    handler = CovarianceRiskHandler()
    params = handler.allocate(spec, horizons=1, n_assets=2)
    cov_factor = np.array([[[2.0, 0.0], [0.0, 3.0]]])
    optimizer_inputs = _inputs(cov_factor=cov_factor)

    handler.update(spec, params, {"optimizer_inputs": optimizer_inputs})

    assert np.array_equal(params["cov_sqrt_0"].value, cov_factor[0])


def test_covariance_risk_handler_computes_psd_sqrt_from_covariance() -> None:
    spec = CovarianceRisk()
    handler = CovarianceRiskHandler()
    params = handler.allocate(spec, horizons=1, n_assets=2)
    covariances = np.array([[[4.0, 0.0], [0.0, 9.0]]])
    optimizer_inputs = _inputs(covariances=covariances)

    handler.update(spec, params, {"optimizer_inputs": optimizer_inputs})

    assert np.allclose(params["cov_sqrt_0"].value, psd_sqrt_factor(covariances[0]))


def test_transaction_cost_handler_updates_coefficients() -> None:
    spec = TransactionCost(
        cost=0.01, pershare_cost=0.1, market_impact=0.2, c_bias=[0.3, -0.4]
    )
    handler = TransactionCostHandler()
    params = handler.allocate(spec, horizons=1, n_assets=2)
    covariances = np.array([[[0.04, 0.0], [0.0, 0.09]]])
    optimizer_inputs = _inputs(covariances=covariances)
    prices = np.array([10.0, 20.0])
    volume = np.array([100.0, 400.0])

    handler.update(
        spec,
        params,
        {"optimizer_inputs": optimizer_inputs, "prices": prices, "volume": volume},
    )

    expected = transaction_cost_coeffs(
        spec,
        n_assets=2,
        prices=prices,
        sigma=np.array([0.2, 0.3]),
        volume=volume,
    )
    assert np.allclose(params["tc_linear"].value, expected[0])
    assert np.allclose(params["tc_impact"].value, expected[1])
    assert np.allclose(params["tc_bias"].value, expected[2])


def test_transaction_cost_handler_requires_covariance_for_market_impact() -> None:
    spec = TransactionCost(market_impact=0.2)
    handler = TransactionCostHandler()
    params = handler.allocate(spec, horizons=1, n_assets=2)

    with pytest.raises(ValueError, match="market_impact requires"):
        handler.update(
            spec, params, {"optimizer_inputs": _inputs(mean=np.array([[0.01, 0.02]]))}
        )


def test_holding_cost_handler_updates_period_rates_and_signs_objective() -> None:
    spec = HoldingCost(
        short_fees=0.2, long_fees=0.1, dividends=0.05, periods_per_year=10
    )
    handler = HoldingCostHandler()
    params = handler.allocate(spec, horizons=1, n_assets=2)

    handler.update(spec, params, {})
    assert params["short_rate"].value == pytest.approx(holding_cost_rates(spec)[0])
    assert params["long_rate"].value == pytest.approx(holding_cost_rates(spec)[1])
    assert params["div_rate"].value == pytest.approx(holding_cost_rates(spec)[2])

    weights = cp.Parameter(2, value=np.array([-0.5, 1.5]))
    expr, constraints = handler.compile(spec, params, weights, weights, horizon=0)
    assert constraints == []
    assert expr.value == pytest.approx(-(0.02 * 0.5) - (0.01 * 1.5) + 0.005 * 1.0)


def test_holding_cost_rates_use_annual_decimal_units() -> None:
    spec = HoldingCost(
        short_fees=0.02,
        long_fees=0.00119,
        dividends=0.01,
        periods_per_year=252,
    )

    assert holding_cost_rates(spec) == pytest.approx(
        (0.02 / 252, 0.00119 / 252, 0.01 / 252)
    )


def test_cvar_risk_handler_updates_scenarios_and_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="CVaR alpha"):
        CVaRRiskHandler().allocate(CVaRRisk(alpha=0.0), 1, 2, 3)

    spec = CVaRRisk(alpha=0.5)
    handler = CVaRRiskHandler()
    params = handler.allocate(spec, horizons=1, n_assets=2, n_scenarios=3)
    scenarios = np.array([[[0.1, 0.0]], [[-0.2, 0.1]], [[0.0, -0.1]]])
    probs = np.array([0.2, 0.3, 0.5])
    optimizer_inputs = _inputs(scenario_returns=scenarios, scenario_probs=probs)

    handler.update(spec, params, {"optimizer_inputs": optimizer_inputs})
    assert np.array_equal(params["probs"].value, probs)
    assert np.array_equal(params["R_0"].value, scenarios[:, 0, :])

    expr, constraints = handler.compile(spec, params, cp.Variable(2), cp.Variable(2), 0)
    assert expr.is_concave()
    assert len(constraints) == 1


def test_cvar_cutting_plane_initializes_anchor_cuts_and_rotates_dynamic_cuts() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        CVaRCuttingPlaneHandler().allocate(CVaRCuttingPlane(max_cuts=2), 1, 2, 3)

    spec = CVaRCuttingPlane(alpha=0.5, max_cuts=4)
    handler = CVaRCuttingPlaneHandler()
    params = handler.allocate(spec, horizons=1, n_assets=2, n_scenarios=3)
    scenarios = np.array([[[0.1, 0.0]], [[-0.2, 0.1]], [[0.0, -0.1]]])
    probs = np.array([0.2, 0.3, 0.5])
    optimizer_inputs = _inputs(scenario_returns=scenarios, scenario_probs=probs)

    handler.update(spec, params, {"optimizer_inputs": optimizer_inputs})
    assert params["cut_count"] == [2]
    assert np.array_equal(params["gw_0"].value[0], np.zeros(2))
    assert params["gz_0"].value[0] == 1.0

    handler._activate_cut(params, 0, np.array([1.0, 2.0]), 0.5, spec.max_cuts)
    handler._activate_cut(params, 0, np.array([3.0, 4.0]), 0.25, spec.max_cuts)
    handler._activate_cut(params, 0, np.array([5.0, 6.0]), 0.125, spec.max_cuts)

    assert params["cut_count"] == [5]
    assert np.array_equal(params["gw_0"].value[2], np.array([5.0, 6.0]))
    assert np.array_equal(params["gw_0"].value[3], np.array([3.0, 4.0]))


def test_cvar_cutting_plane_refine_adds_cut_when_gap_is_material() -> None:
    spec = CVaRCuttingPlane(alpha=0.5, max_cuts=4, tol=1e-12)
    handler = CVaRCuttingPlaneHandler()
    params = handler.allocate(spec, horizons=1, n_assets=2, n_scenarios=3)
    scenarios = np.array([[[0.1, 0.0]], [[-0.2, 0.1]], [[0.0, -0.1]]])
    probs = np.array([0.2, 0.3, 0.5])
    optimizer_inputs = _inputs(scenario_returns=scenarios, scenario_probs=probs)
    handler.update(spec, params, {"optimizer_inputs": optimizer_inputs})
    params["zeta_0"].value = 0.0
    params["theta_0"].value = -1.0

    converged = handler.refine(spec, params, np.array([[1.0, 0.0]]), optimizer_inputs)

    assert converged is False
    assert params["cut_count"] == [3]
