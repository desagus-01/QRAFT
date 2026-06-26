from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qraft.backtest.execution import BacktestResult
from qraft.core import metrics


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    cvar: float
    hit_rate: float
    avg_turnover: float
    total_cost: float
    n_periods: int
    n_warnings: int
    n_solver_failures: int
    held_fraction: float

    @classmethod
    def from_backtest(
        cls,
        result: BacktestResult,
        *,
        periods_per_year: float = 252.0,
        risk_free_rate: float = 0.0,
        active_only: bool = True,
    ) -> PerformanceSummary:
        nav = np.asarray(result.nav, dtype=float)
        if active_only and result.periods:
            start = result.nav_dates.index(result.periods[0].execution_bar)
            nav = nav[start:]
        rets = metrics.returns_from_nav(nav)
        failures = sum(
            1
            for p in result.periods
            if p.solver_status == "solver_error" or p.decision_error is not None
        )
        return cls(
            total_return=float(nav[-1] / nav[0] - 1.0) if nav.size else 0.0,
            annualised_return=metrics.annualised_return(nav, periods_per_year),
            annualised_vol=metrics.annualised_vol(rets, periods_per_year),
            sharpe=metrics.sharpe(
                rets, risk_free_rate / periods_per_year, periods_per_year
            ),
            sortino=metrics.sortino(
                rets, risk_free_rate / periods_per_year, periods_per_year
            ),
            max_drawdown=metrics.max_drawdown(nav),
            calmar=metrics.calmar(nav, periods_per_year),
            cvar=float(
                metrics.cvar(rets, prob=None, alpha=0.05, distribution_type="pnl")
            )
            if rets.size >= 2
            else 0.0,
            hit_rate=float(np.mean(rets > 0)) if rets.size else 0.0,
            avg_turnover=float(np.mean(result.period_turnovers))
            if len(result.period_turnovers)
            else 0.0,
            total_cost=result.total_cost,
            n_periods=len(result.periods),
            n_warnings=len(result.warnings_log),
            n_solver_failures=failures,
            held_fraction=failures / len(result.periods) if result.periods else 0.0,
        )

    def to_dict(self) -> dict[str, float]:
        return {f: getattr(self, f) for f in self.__slots__}
