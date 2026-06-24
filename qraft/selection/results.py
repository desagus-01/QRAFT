from __future__ import annotations

from dataclasses import dataclass

from qraft.backtest.execution import BacktestResult
from qraft.backtest.metrics import PerformanceSummary


@dataclass(frozen=True, slots=True)
class CandidateFailure:
    risk_aversion: float
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateResult:
    risk_aversion: float
    summary: PerformanceSummary
    backtest: BacktestResult | None = None
    failure: CandidateFailure | None = None


@dataclass(frozen=True, slots=True)
class SelectionReport:
    candidates: tuple[CandidateResult, ...]
    selected_risk_aversion: float | None
    rule: str
