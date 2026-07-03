from datetime import datetime

import numpy as np
import polars as pl
import pytest

from qraft import Backtest, ForecastSource, Validation
from qraft.backtest.configs import (
    BacktestConfig,
    CombinatorialCVConfig,
    WalkForwardConfig,
)
from qraft.backtest.inputs import PrecomputedInputsProvider
from qraft.backtest.metrics import PerformanceSummary
from qraft.core.market import MarketData
from qraft.backtest.selection import __all__ as selection_all
from qraft.backtest.selection.combinatorial import CombinatorialReport
from qraft.backtest.selection.evaluation import CandidateEvaluation
from qraft.backtest.selection.results import CandidateResult, SelectionReport
from qraft.backtest.selection.splits import Fold
from qraft.backtest.selection.validation import Validation as SelectionValidation
from qraft.backtest.selection.walkforward import FoldResult, WalkForwardReport
from qraft.backtest.simulator import run_backtest
from qraft.construction.optimization.inputs import InputPlan, PolicyInputs
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
                "A": [10.0, 11.0, 12.0, 13.0],
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
        t: PolicyInputs.from_arrays(
            assets=["A"], mean=np.ones((1, 1)), cash_return=np.array([0.0])
        )
        for t in market.trading_bars[:-1]
    }

    facade = Backtest(market, policy, source=source, config=config).run()
    manual = run_backtest(
        market,
        policy,
        schedule=config.schedule,
        inputs=PrecomputedInputsProvider(source),
        initial_cash=config.initial_cash,
    )

    np.testing.assert_allclose(facade.nav, manual.nav)


def test_backtest_rejects_current_only_views():
    market = _market().with_current_views(_NoopView())
    policy = EqualWeightPolicy(target_cash_weight=0.25)

    with pytest.raises(ValueError, match="current-only views"):
        Backtest(market=market, policy=policy).run()


def test_validation_runs_explicit_reports(monkeypatch):
    market = _market()
    policy = EqualWeightPolicy(target_cash_weight=0.0)
    seen = {}

    def fake_walk_forward(self, cfg):
        seen["walk"] = cfg
        return _FakeWalkReport()

    def fake_combinatorial(self, cfg):
        seen["cpcv"] = cfg
        return _FakeCpcvReport()

    monkeypatch.setattr(
        "qraft.backtest.selection.validation.Validation.walk_forward", fake_walk_forward
    )
    monkeypatch.setattr(
        "qraft.backtest.selection.validation.Validation.combinatorial",
        fake_combinatorial,
    )

    validation = SelectionValidation(market, policy, {}, source={}, plan=InputPlan())

    assert isinstance(validation.walk_forward(WalkForwardConfig()), _FakeWalkReport)
    assert isinstance(validation.combinatorial(), _FakeCpcvReport)

    assert isinstance(seen["walk"], WalkForwardConfig)
    assert isinstance(seen["cpcv"], CombinatorialCVConfig)


def test_validation_rejects_current_only_views():
    market = _market().with_current_views(_NoopView())
    policy = EqualWeightPolicy(target_cash_weight=0.0)

    with pytest.raises(ValueError, match="current-only views"):
        SelectionValidation(
            market, policy, {}, source={}, plan=InputPlan()
        ).walk_forward(WalkForwardConfig())


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

    validation = SelectionValidation(market, policy, {}, source={}, plan=InputPlan())

    assert isinstance(validation.walk_forward(WalkForwardConfig()), _FakeWalkReport)
    assert isinstance(
        validation.combinatorial(CombinatorialCVConfig()), _FakeCpcvReport
    )
    assert len(calls) == 1


def test_validation_tune_requires_oos_report() -> None:
    validation = SelectionValidation(
        _market(),
        EqualWeightPolicy(target_cash_weight=0.0),
        {},
        source={},
        plan=InputPlan(),
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

    result = SelectionValidation(
        _market(), policy, {}, source={}, plan=InputPlan()
    ).tune(report, cfg=WalkForwardConfig(), metric="total_return")

    assert result.report is report
    assert result.selected_params == params
    assert isinstance(result.selected_policy, MPOPolicy)


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
    assert ForecastSource is not None
    assert WalkForwardReport is not None
    assert CombinatorialReport is not None
    assert "combinatorial_purged" in selection_all
    assert "run_combinatorial_purged" not in selection_all
