from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from qraft.backtest.result import BacktestResult, PerformanceSummary
from qraft.backtest.result.result import plot_nav
from qraft.construction.policies import PolicyProtocol


@dataclass(frozen=True, slots=True)
class PolicyParams:
    """Immutable candidate hyperparameter values."""

    values: tuple[tuple[str, Any], ...]

    @classmethod
    def of(cls, **kwargs: Any) -> "PolicyParams":
        return cls(tuple(sorted(kwargs.items())))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def __bool__(self) -> bool:
        return bool(self.values)

    def __str__(self) -> str:
        return ", ".join(_format_param(k, v) for k, v in self.values)


def _format_param(key: str, value: Any) -> str:
    if isinstance(value, float):
        return f"{key}={value:g}"
    return f"{key}={value}"


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    params: PolicyParams
    policy: PolicyProtocol


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

    @property
    def selected(self) -> CandidateResult | None:
        if self.selected_params is None:
            return None
        for c in self.candidates:
            if c.params == self.selected_params:
                return c
        return None

    def metrics_df(self) -> pl.DataFrame:
        rows = []
        for c in self.candidates:
            if c.summary is None:
                continue
            row = c.summary.to_dict()
            row["params"] = str(c.params)
            row["selected"] = c.params == self.selected_params
            rows.append(row)
        return pl.DataFrame(rows)

    def plot(self) -> None:
        results = [c.backtest for c in self.candidates if c.backtest is not None]
        labels = [str(c.params) for c in self.candidates if c.backtest is not None]
        fig = plot_nav(results, labels=labels)
        fig.show()
