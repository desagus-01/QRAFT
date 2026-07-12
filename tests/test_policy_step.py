from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import cvxpy as cp
import numpy as np

from qraft.backtest.engine.policy_step import decide_or_hold
from qraft.backtest.engine.schedule import DecisionPoint
from qraft.construction.optimization.inputs import RequiredPolicyInputs
from qraft.construction.policies.policy_decision import PolicyDecision
from qraft.construction.state import PortfolioState


@dataclass
class _RaisingPolicy:
    _exc: Exception
    _name: str = "raising"

    @property
    def name(self) -> str:
        return self._name

    @property
    def min_history(self) -> int:
        return 0

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: Any = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        raise self._exc

    def required_inputs(self) -> RequiredPolicyInputs:
        return RequiredPolicyInputs()


@dataclass
class _HappyPolicy:
    _name: str = "happy"

    @property
    def name(self) -> str:
        return self._name

    @property
    def min_history(self) -> int:
        return 0

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: Any = None,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            asset_order=["A"],
            target_weights_risk=np.array([0.5]),
            target_cash_weight=0.5,
        )

    def required_inputs(self) -> RequiredPolicyInputs:
        return RequiredPolicyInputs()


def _state() -> PortfolioState:
    return PortfolioState(
        asset_order=["A"],
        initial_prices=np.array([100.0]),
        shares=np.array([1.0]),
        cash=100.0,
    )


def _point() -> DecisionPoint:
    snapshot = MagicMock()
    return DecisionPoint(
        index=0,
        decision_bar=datetime(2024, 1, 31),
        execution_bar=datetime(2024, 2, 1),
        snapshot=snapshot,
    )


def test_solver_error_degrades_to_hold() -> None:
    policy = _RaisingPolicy(cp.error.SolverError("CLARABEL failed"))
    result = decide_or_hold(policy, None, _state(), _point(), ["A"])

    assert result.decision.hold is True
    assert result.decision.asset_order == ["A"]
    assert result.solver_status == "solver_error"
    assert result.warning is not None
    assert "SolverError" in result.decision_error


def test_runtime_error_degrades_to_hold() -> None:
    policy = _RaisingPolicy(RuntimeError("cutting-plane nonconverged"))
    result = decide_or_hold(policy, None, _state(), _point(), ["A"])

    assert result.decision.hold is True
    assert result.solver_status == "solver_error"
    assert result.warning is not None
    assert "RuntimeError" in result.decision_error


def test_value_error_degrades_to_hold() -> None:
    policy = _RaisingPolicy(ValueError("bad input"))
    result = decide_or_hold(policy, None, _state(), _point(), ["A"])

    assert result.decision.hold is True
    assert result.warning is not None


def test_successful_decision_returns_result() -> None:
    policy = _HappyPolicy()
    result = decide_or_hold(policy, None, _state(), _point(), ["A"])

    assert result.decision.hold is False
    assert result.warning is None
    assert result.decision_error is None
    np.testing.assert_allclose(result.decision.target_weights_risk, [0.5])
