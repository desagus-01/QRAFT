import numpy as np
import pytest

from qraft.construction.frontier import (
    FrontierKind,
    MPOFrontierConfig,
    MPOFrontierRunner,
    ex_ante_metrics,
    ex_post_terminal_cvar,
)
from qraft.construction.optimization.constraints import (
    FullyInvested,
    LongOnly,
    MinCashWeight,
)
from qraft.construction.optimization.moments import HorizonMoments, MomentsConfig
from qraft.construction.optimization.optimization import MPOResult, MultiPeriodOptimizer
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths


def _cash_file(tmp_path) -> str:
    path = tmp_path / "cash.csv"
    path.write_text("date,rate\n2026-01-01,0.0\n")
    return str(path)


def _forecast_paths() -> ForecastPaths:
    initial_prices = {"A": 100.0, "B": 100.0}
    returns = {
        "A": np.array(
            [
                [0.08, 0.08],
                [0.06, 0.06],
                [-0.02, -0.02],
                [0.00, 0.00],
            ]
        ),
        "B": np.array(
            [
                [0.020, 0.020],
                [0.025, 0.025],
                [0.015, 0.015],
                [0.020, 0.020],
            ]
        ),
    }
    asset_paths = {}
    for asset, ret in returns.items():
        prices = np.empty_like(ret)
        prices[:, 0] = initial_prices[asset] * (1.0 + ret[:, 0])
        prices[:, 1] = prices[:, 0] * (1.0 + ret[:, 1])
        asset_paths[asset] = prices

    return ForecastPaths(
        asset_paths=asset_paths,
        path_probs=np.full(4, 0.25),
        initial_prices=initial_prices,
        universe=AssetUniverse.factors_free(["A", "B"]),
    )


def _state(forecasts: ForecastPaths) -> PortfolioState:
    return PortfolioState.from_cash(
        cash=1_000.0,
        assets=["A", "B"],
        asset_forecasts=forecasts,
    )


def _frontier_config(*, risk_aversions=(0.0, 20.0)) -> MPOFrontierConfig:
    return MPOFrontierConfig(
        risk_aversions=risk_aversions,
        transaction_cost_weight=0.0,
        constraints=(LongOnly(), FullyInvested()),
        solver_options={"solver": "CLARABEL"},
    )


def _moments_config(tmp_path) -> MomentsConfig:
    return MomentsConfig(
        cash_path=_cash_file(tmp_path),
        horizons=2,
        expectation_tolerance=None,
    )


def _simple_moments() -> HorizonMoments:
    return HorizonMoments(
        assets=["A", "B"],
        correlations=np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
        covariances=np.array(
            [
                [[0.04, 0.0], [0.0, 0.09]],
                [[0.01, 0.0], [0.0, 0.04]],
            ]
        ),
        mean=np.array([[0.01, 0.02], [0.03, 0.04]]),
        scenario_returns=np.zeros((2, 2, 2)),
        scenario_probs=np.array([0.5, 0.5]),
        cash_return=np.array([0.001, 0.002]),
    )


def _simple_result() -> MPOResult:
    return MPOResult(
        assets=["A", "B"],
        planned_weights=np.array([[0.6, 0.3], [0.4, 0.5]]),
        planned_trades=np.zeros((2, 2)),
        planned_cash=np.array([0.1, 0.1]),
        planned_cash_trades=np.zeros(2),
        initial_weights=np.array([0.0, 0.0]),
        initial_cash=1.0,
        status="optimal",
        objective_value=0.0,
        solver_stats=None,
    )


def test_ex_ante_metrics_are_analytic_sum_of_horizon_values() -> None:
    metrics = ex_ante_metrics(_simple_result(), _simple_moments())

    assert metrics["expected_return"] == pytest.approx(
        0.6 * 0.01 + 0.3 * 0.02 + 0.4 * 0.03 + 0.5 * 0.04 + 0.1 * 0.001 + 0.1 * 0.002
    )
    assert metrics["volatility"] == pytest.approx(
        np.sqrt(
            np.array([0.6, 0.3])
            @ np.array([[0.04, 0.0], [0.0, 0.09]])
            @ np.array([0.6, 0.3])
            + np.array([0.4, 0.5])
            @ np.array([[0.01, 0.0], [0.0, 0.04]])
            @ np.array([0.4, 0.5])
        )
    )


def test_ex_post_terminal_cvar_matches_hand_computed_tail() -> None:
    moments = HorizonMoments(
        assets=["A", "B"],
        correlations=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        covariances=np.array([[[0.01, 0.0], [0.0, 0.01]]]),
        mean=np.array([[0.0, 0.0]]),
        scenario_returns=np.array(
            [[[-0.20, -0.20]], [[-0.10, -0.10]], [[0.05, 0.05]], [[0.10, 0.10]]]
        ),
        scenario_probs=np.full(4, 0.25),
        cash_return=np.array([0.0]),
    )
    result = MPOResult(
        assets=["A", "B"],
        planned_weights=np.array([[0.5, 0.5]]),
        planned_trades=np.zeros((1, 2)),
        planned_cash=np.array([0.0]),
        planned_cash_trades=np.zeros(1),
        initial_weights=np.array([0.0, 0.0]),
        initial_cash=1.0,
        status="optimal",
        objective_value=0.0,
        solver_stats=None,
    )

    assert ex_post_terminal_cvar(result, moments, alpha=0.5) == pytest.approx(0.15)


def test_run_single_returns_labelled_grid_and_computes_moments_once(
    tmp_path,
    monkeypatch,
) -> None:
    calls = 0
    original = HorizonMoments.from_forecast_paths

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(HorizonMoments, "from_forecast_paths", spy)

    forecasts = _forecast_paths()
    runner = MPOFrontierRunner(_frontier_config(), _moments_config(tmp_path))
    result = runner.run_single(_state(forecasts), forecasts)

    assert calls == 1
    assert result.kind is FrontierKind.EX_ANTE_MPO
    assert len(result.points) == 2
    assert all(point.metric_kind is FrontierKind.EX_ANTE_MPO for point in result.points)
    assert all(
        point.status in {"optimal", "optimal_inaccurate"} for point in result.points
    )


def test_optimizer_cache_is_reused_across_same_universe_dates(
    tmp_path,
    monkeypatch,
) -> None:
    init_count = 0
    original_init = MultiPeriodOptimizer.__init__

    def spy_init(self, *args, **kwargs):
        nonlocal init_count
        init_count += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(MultiPeriodOptimizer, "__init__", spy_init)

    forecasts = _forecast_paths()
    runner = MPOFrontierRunner(_frontier_config(), _moments_config(tmp_path))

    first, second = runner.run_repeated(
        [
            (_state(forecasts), forecasts),
            (_state(forecasts), forecasts),
        ]
    )
    assert len(first.points) == 2
    assert len(second.points) == 2
    assert init_count == 2


def test_volatility_weakly_decreases_as_gamma_increases(tmp_path) -> None:
    forecasts = _forecast_paths()
    config = _frontier_config(risk_aversions=(0.0, 10.0, 100.0))
    runner = MPOFrontierRunner(config, _moments_config(tmp_path))
    result = runner.run_single(_state(forecasts), forecasts)

    valid_points = [point for point in result.points if point.volatility is not None]
    vols = [point.volatility for point in valid_points]

    assert vols == sorted(vols, reverse=True)


def test_forced_infeasible_point_is_recorded_without_aborting(tmp_path) -> None:
    forecasts = _forecast_paths()
    config = MPOFrontierConfig(
        risk_aversions=(1.0,),
        transaction_cost_weight=0.0,
        constraints=(FullyInvested(), MinCashWeight(0.1)),
        solver_options={"solver": "CLARABEL"},
    )
    runner = MPOFrontierRunner(config, _moments_config(tmp_path))

    result = runner.run_single(_state(forecasts), forecasts)

    assert len(result.points) == 1
    assert result.points[0].status in {"infeasible", "infeasible_inaccurate"}
    assert result.points[0].failure_message
