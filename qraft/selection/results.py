from __future__ import annotations

from dataclasses import dataclass

from qraft.backtest.execution import BacktestResult
from qraft.backtest.metrics import PerformanceSummary


@dataclass(frozen=True, slots=True)
class PolicyParams:
    """
    Hyperparameter class for backtesting
    """

    values: tuple[tuple[str, float], ...]

    @classmethod
    def of(cls, **kwargs: float):
        return cls(tuple(sorted(kwargs.items())))

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    @property
    def risk_aversion(self) -> float:
        return self.as_dict()["risk_aversion"]

    def __str__(self) -> str:
        return ", ".join(f"{k}={v:g}" for k, v in self.values)


@dataclass(frozen=True, slots=True)
class CandidateFailure:
    params: PolicyParams
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateResult:
    params: PolicyParams
    summary: PerformanceSummary | None = None
    backtest: BacktestResult | None = None
    failure: CandidateFailure | None = None


@dataclass(frozen=True, slots=True)
class SelectionReport:
    candidates: tuple[CandidateResult, ...]
    selected_params: PolicyParams | None
    rule: str
