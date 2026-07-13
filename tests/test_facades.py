from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft import Backtest, ForecastSpec, Validation
from qraft.backtest.configs import (
    BacktestConfig,
    CombinatorialCVConfig,
    WalkForwardConfig,
)
from qraft.backtest.result import BacktestResult
from qraft.backtest.inputs import PrecomputedInputsProvider
from qraft.backtest.result import PerformanceSummary
from qraft.core.market import MarketData
from qraft.backtest.selection import __all__ as selection_all
from qraft.backtest.selection.candidate_eval import CandidateEvaluation
from qraft.backtest.selection.reports import (
    CombinatorialReport,
    FoldResult,
    WalkForwardReport,
)
from qraft.backtest.selection.results import CandidateResult, SelectionReport
from qraft.backtest.selection.splits import Fold
from qraft.backtest.selection.validation import Validation as SelectionValidation
from qraft.backtest.engine.loop import run_backtest
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.policies import EqualWeightPolicy, MPOPolicy
from qraft.core.schedule import RebalanceSchedule
from qraft.forecast.forecast_paths import AssetUniverse
from qraft.backtest.selection.results import PolicyParams
from qraft.backtest.selection.validation import ValidationResult


class _NoopView:
    def apply(self, panel):
        return panel


class _FakeWalkReport:
    folds = ()


class _FakeCpcvReport:
    selected_params = None


def _market() -> MarketData:
    return MarketData.from_prices(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, day) for day in range(1, 5)],
                "A": [20.0, 22.0, 24.0, 26.0],
            }
        ),
        AssetUniverse.factors_free(["A"]),
    )


def test_backtest_facade_matches_manual_run_backtest():
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.25)
    config = BacktestConfig(schedule=RebalanceSchedule("every_bar"), initial_cash=100.0)

    facade = Backtest(market=market, policy=policy, config=config).run()
    manual = run_backtest(
        market,
        policy,
        schedule=config.schedule,
        initial_cash=config.initial_cash,
    )

    assert facade.periods == manual.periods
    assert facade.nav_dates == manual.nav_dates
    np.testing.assert_allclose(facade.nav, manual.nav)


def test_backtest_facade_accepts_precomputed_source():
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0)
    config = BacktestConfig(schedule=RebalanceSchedule("every_bar"), initial_cash=100.0)
    source = {
        t: OptimizerInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
        )
        for t in market.trading_bars[:-1]
    }

    facade = Backtest(market, policy, forecasts=source, config=config).run()
    manual = run_backtest(
        market,
        policy,
        schedule=config.schedule,
        inputs=PrecomputedInputsProvider(source),
        initial_cash=config.initial_cash,
        config=config,
    )

    np.testing.assert_allclose(facade.nav, manual.nav)


def test_backtest_retains_optimizer_inputs_when_configured() -> None:
    market = _market()
    config = BacktestConfig(
        schedule=RebalanceSchedule("every_bar"),
        retain_optimizer_inputs=True,
    )
    source = {
        t: OptimizerInputs(
            assets=["A"],
            mean=np.ones((1, 1)),
            covariances=np.ones((1, 1, 1)),
            scenario_returns=np.ones((2, 1, 1)),
            scenario_probs=np.array([0.5, 0.5]),
        )
        for t in market.trading_bars[:-1]
    }

    result = Backtest(
        market,
        EqualWeightPolicy(target_cash_weight=0.0),
        forecasts=source,
        config=config,
    ).run()

    retained = result.periods[0].optimizer_inputs
    assert retained is not None
    np.testing.assert_allclose(retained.mean, np.ones((1, 1)))
    np.testing.assert_allclose(retained.covariances, np.ones((1, 1, 1)))
    assert retained.scenario_returns is None
    assert retained.scenario_probs is None


def test_backtest_does_not_retain_optimizer_inputs_by_default() -> None:
    market = _market()
    source = {
        t: OptimizerInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
        )
        for t in market.trading_bars[:-1]
    }

    result = Backtest(
        market,
        EqualWeightPolicy(target_cash_weight=0.0),
        forecasts=source,
        config=BacktestConfig(schedule=RebalanceSchedule("every_bar")),
    ).run()

    assert result.periods[0].optimizer_inputs is None


def test_backtest_retains_optimizer_scenarios_when_configured() -> None:
    market = _market()
    config = BacktestConfig(
        schedule=RebalanceSchedule("every_bar"),
        retain_optimizer_inputs=True,
        retain_optimizer_scenarios=True,
    )
    source = {
        t: OptimizerInputs(
            assets=["A"],
            mean=np.ones((1, 1)),
            scenario_returns=np.ones((2, 1, 1)),
            scenario_probs=np.array([0.5, 0.5]),
        )
        for t in market.trading_bars[:-1]
    }

    result = Backtest(
        market,
        EqualWeightPolicy(target_cash_weight=0.0),
        forecasts=source,
        config=config,
    ).run()

    retained = result.periods[0].optimizer_inputs
    assert retained is not None
    np.testing.assert_allclose(retained.scenario_returns, np.ones((2, 1, 1)))
    np.testing.assert_allclose(retained.scenario_probs, np.array([0.5, 0.5]))


def test_backtest_facade_empty_precompute_table_means_no_inputs(monkeypatch):
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0)
    config = BacktestConfig(schedule=RebalanceSchedule("every_bar"), initial_cash=100.0)

    monkeypatch.setattr("qraft.backtest.backtest.precompute_inputs", lambda *a, **k: {})

    result = Backtest(market, policy, forecasts=[], config=config).run()

    assert result.periods


def test_validation_runs_explicit_reports(monkeypatch):
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0)
    seen = {}

    def fake_walk_forward(self, cfg):
        seen["walk"] = cfg
        return _FakeWalkReport()

    def fake_combinatorial(self, cfg=None):
        if cfg is None:
            cfg = CombinatorialCVConfig()
        seen["cpcv"] = cfg
        return _FakeCpcvReport()

    monkeypatch.setattr(
        "qraft.backtest.selection.validation.Validation.walk_forward", fake_walk_forward
    )
    monkeypatch.setattr(
        "qraft.backtest.selection.validation.Validation.combinatorial",
        fake_combinatorial,
    )

    validation = SelectionValidation(market, policy, {}, forecasts={})

    assert isinstance(validation.walk_forward(WalkForwardConfig()), _FakeWalkReport)
    assert isinstance(validation.combinatorial(), _FakeCpcvReport)

    assert isinstance(seen["walk"], WalkForwardConfig)
    assert isinstance(seen["cpcv"], CombinatorialCVConfig)


def test_validation_reuses_candidate_evaluation(monkeypatch):
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0)
    evaluation = CandidateEvaluation(
        candidate_results=(),
        dates=[],
        backtest_config=BacktestConfig(),
    )
    calls = []

    def fake_evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return evaluation

    def fake_walk_from_evaluation(evaluation_arg, **kwargs):
        assert evaluation_arg is evaluation
        return _FakeWalkReport()

    def fake_cpcv_from_evaluation(evaluation_arg, **kwargs):
        assert evaluation_arg is evaluation
        return _FakeCpcvReport()

    monkeypatch.setattr(
        "qraft.backtest.selection.validation.evaluate_candidate_grid", fake_evaluate
    )
    monkeypatch.setattr(
        "qraft.backtest.selection.validation.walk_forward_from_evaluation",
        fake_walk_from_evaluation,
    )
    monkeypatch.setattr(
        "qraft.backtest.selection.validation.combinatorial_from_evaluation",
        fake_cpcv_from_evaluation,
    )

    validation = SelectionValidation(
        market,
        policy,
        {},
        forecasts={},
        backtest_config=BacktestConfig(schedule=RebalanceSchedule("every_bar")),
    )

    assert isinstance(
        validation.walk_forward(WalkForwardConfig(train_size=1, test_size=2)),
        _FakeWalkReport,
    )
    assert isinstance(
        validation.combinatorial(CombinatorialCVConfig(n_groups=2, n_test_groups=1)),
        _FakeCpcvReport,
    )
    assert len(calls) == 1


def test_validation_combinatorial_checks_feasibility_before_evaluation(monkeypatch):
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0, min_history=1)
    calls = []

    def fake_evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("grid evaluation should not run")

    monkeypatch.setattr(
        "qraft.backtest.selection.validation.evaluate_candidate_grid", fake_evaluate
    )
    validation = SelectionValidation(
        market,
        policy,
        {},
        forecasts={},
        backtest_config=BacktestConfig(schedule=RebalanceSchedule("every_bar")),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"CPCV needs >= n_groups=50 decision dates; schedule 'every_bar' "
            r"\+ min_history=1 yields 3"
        ),
    ):
        validation.combinatorial(CombinatorialCVConfig(n_groups=50))

    assert calls == []


def test_validation_walk_forward_checks_feasibility_before_evaluation(monkeypatch):
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0, min_history=1)
    calls = []

    def fake_evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("grid evaluation should not run")

    monkeypatch.setattr(
        "qraft.backtest.selection.validation.evaluate_candidate_grid", fake_evaluate
    )
    validation = SelectionValidation(
        market,
        policy,
        {},
        forecasts={},
        backtest_config=BacktestConfig(schedule=RebalanceSchedule("every_bar")),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Walk-forward needs >= train_size=3 \+ embargo=0 \+ "
            r"test_size=2 \(5\) decision dates; schedule 'every_bar' "
            r"\+ min_history=1 yields 3"
        ),
    ):
        validation.walk_forward(WalkForwardConfig(train_size=3, test_size=2))

    assert calls == []


def test_validation_tune_requires_oos_report() -> None:
    validation = SelectionValidation(
        _market(),
        EqualWeightPolicy(target_cash_weight=0.0),
        {},
        forecasts={},
    )

    with pytest.raises(TypeError, match="missing.*report"):
        validation.tune()  # type: ignore[call-arg]


def test_validation_tune_returns_result_with_report() -> None:
    policy = MPOPolicy.preset("mean_covariance", name="template")
    dates = [datetime(2024, 1, day) for day in range(1, 5)]
    params = PolicyParams.of(risk_aversion=2.0)
    candidate_backtest = run_backtest(
        _market(), EqualWeightPolicy(target_cash_weight=0.0)
    )
    candidate = CandidateResult(
        params=params,
        summary=PerformanceSummary.from_backtest(candidate_backtest),
        backtest=candidate_backtest,
    )
    evaluation = CandidateEvaluation(
        candidate_results=(candidate,),
        dates=dates,
        backtest_config=BacktestConfig(periods_per_year=252.0),
    )
    report = WalkForwardReport(
        folds=(
            FoldResult(
                fold=Fold(train=(dates[0], dates[1]), test=(dates[2], dates[3])),
                selection=SelectionReport(
                    candidates=(candidate,), selected_params=params, rule="test"
                ),
                oos_summary=None,
            ),
        ),
        oos_summary=None,
        oos_nav_dates=[],
        oos_nav=np.array([], dtype=float),
        evaluation=evaluation,
    )

    result = SelectionValidation(_market(), policy, {}, forecasts={}).tune(
        report, cfg=WalkForwardConfig(), score="total_return"
    )

    assert result.report is report
    assert result.selected_params == params
    assert isinstance(result.selected_policy, MPOPolicy)
    assert result.selected_result is candidate
    assert result.backtest is candidate_backtest


def test_validation_result_accessors_raise_without_selection() -> None:
    result = ValidationResult(
        report=WalkForwardReport(
            folds=(),
            oos_summary=None,
            oos_nav_dates=[],
            oos_nav=np.array([], dtype=float),
            evaluation=CandidateEvaluation(
                candidate_results=(),
                dates=[],
                backtest_config=BacktestConfig(),
            ),
        ),
        base_policy=EqualWeightPolicy(target_cash_weight=0.0),
        selected_params=None,
    )

    with pytest.raises(ValueError, match="did not select"):
        result.selected_result

    with pytest.raises(ValueError, match="did not select"):
        result.backtest


def test_validation_result_backtest_raises_when_candidate_has_no_backtest() -> None:
    params = PolicyParams.of(risk_aversion=2.0)
    candidate = CandidateResult(params=params)
    result = ValidationResult(
        report=WalkForwardReport(
            folds=(),
            oos_summary=None,
            oos_nav_dates=[],
            oos_nav=np.array([], dtype=float),
            evaluation=CandidateEvaluation(
                candidate_results=(candidate,),
                dates=[],
                backtest_config=BacktestConfig(),
            ),
        ),
        base_policy=EqualWeightPolicy(target_cash_weight=0.0),
        selected_params=params,
    )

    assert result.selected_result is candidate
    with pytest.raises(ValueError, match="does not have a backtest"):
        result.backtest


def test_validation_tune_defaults_to_most_selected_not_oos_argmax() -> None:
    policy = MPOPolicy.preset("mean_covariance", name="template")
    dates = [datetime(2024, 1, day) for day in range(1, 6)]
    stable = PolicyParams.of(risk_aversion=1.0)
    lucky = PolicyParams.of(risk_aversion=9.0)
    stable_backtest = BacktestResult(
        policy_name="stable",
        asset_order=[],
        nav_dates=dates,
        nav=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
        periods=[],
    )
    lucky_backtest = BacktestResult(
        policy_name="lucky",
        asset_order=[],
        nav_dates=dates,
        nav=np.array([100.0, 99.0, 98.0, 120.0, 150.0]),
        periods=[],
    )
    candidates = (
        CandidateResult(params=stable, backtest=stable_backtest),
        CandidateResult(params=lucky, backtest=lucky_backtest),
    )
    report = WalkForwardReport(
        folds=(
            FoldResult(
                fold=Fold(train=(dates[0], dates[1]), test=(dates[2], dates[3])),
                selection=SelectionReport(
                    candidates=candidates, selected_params=stable, rule="train"
                ),
                oos_summary=None,
            ),
            FoldResult(
                fold=Fold(train=(dates[1], dates[2]), test=(dates[3], dates[4])),
                selection=SelectionReport(
                    candidates=candidates, selected_params=stable, rule="train"
                ),
                oos_summary=None,
            ),
            FoldResult(
                fold=Fold(train=(dates[0], dates[2]), test=(dates[3], dates[4])),
                selection=SelectionReport(
                    candidates=candidates, selected_params=lucky, rule="train"
                ),
                oos_summary=None,
            ),
        ),
        oos_summary=None,
        oos_nav_dates=[],
        oos_nav=np.array([], dtype=float),
        evaluation=CandidateEvaluation(
            candidate_results=candidates,
            dates=dates,
            backtest_config=BacktestConfig(periods_per_year=252.0),
        ),
    )

    result = SelectionValidation(_market(), policy, {}, forecasts={}).tune(
        report, cfg=WalkForwardConfig(), score="total_return"
    )

    assert result.selected_params == stable


def test_walk_forward_report_require_rejects_high_pbo() -> None:
    report = WalkForwardReport(
        folds=(),
        oos_summary=None,
        oos_nav_dates=[],
        oos_nav=np.array([], dtype=float),
        pbo=0.944,
    )

    with pytest.raises(ValueError, match="PBO"):
        report.require(max_pbo=0.2)


def test_validation_result_selected_policy_applies_selected_params():
    policy = MPOPolicy.preset("mean_covariance", name="template")
    result = ValidationResult(
        report=_FakeWalkReport(),
        base_policy=policy,
        selected_params=PolicyParams.of(risk_aversion=2.0),
    )

    selected = result.selected_policy

    assert isinstance(selected, MPOPolicy)
    assert selected.problem.objective.terms[2].weight == 2.0


def test_validation_result_selected_policy_raises_without_selection():
    result = ValidationResult(
        report=_FakeWalkReport(),
        base_policy=EqualWeightPolicy(target_cash_weight=0.0),
        selected_params=None,
    )

    with pytest.raises(ValueError, match="did not select"):
        result.selected_policy


def test_public_facade_exports():
    assert Backtest.__name__ == "Backtest"
    assert Validation.__name__ == "Validation"
    assert ForecastSpec is not None
    assert WalkForwardReport is not None
    assert CombinatorialReport is not None
    assert "combinatorial_purged" in selection_all
    assert "run_combinatorial_purged" not in selection_all
