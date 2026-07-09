from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from qraft.core import metrics
from qraft.core.cadence import infer_periods_per_year


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
        result,
        *,
        periods_per_year: float | None = None,
        risk_free_rate: float = 0.0,
        active_only: bool = True,
    ) -> PerformanceSummary:
        nav = np.asarray(result.nav, dtype=float)
        dates = result.nav_dates
        if active_only and result.periods:
            start = result.nav_dates.index(result.periods[0].execution_bar)
            nav = nav[start:]
            dates = dates[start:]
        failures = sum(
            1
            for p in result.periods
            if p.solver_status == "solver_error" or p.decision_error is not None
        )
        return summary_from_nav(
            nav,
            dates=dates,
            periods_per_year=periods_per_year or result.periods_per_year,
            risk_free_rate=risk_free_rate,
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


def summary_from_nav(
    nav: NDArray[np.floating],
    *,
    dates: Sequence[datetime] | None = None,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
    avg_turnover: float = 0.0,
    total_cost: float = 0.0,
    n_periods: int | None = None,
    n_warnings: int = 0,
    n_solver_failures: int = 0,
    held_fraction: float = 0.0,
) -> PerformanceSummary:
    nav = np.asarray(nav, dtype=float)
    resolved_periods_per_year = periods_per_year
    if resolved_periods_per_year is None and dates is not None and len(dates) >= 2:
        resolved_periods_per_year = infer_periods_per_year(dates)
    if resolved_periods_per_year is None:
        raise ValueError("periods_per_year is required when it cannot be inferred.")

    rets = metrics.returns_from_nav(nav)
    return PerformanceSummary(
        total_return=float(nav[-1] / nav[0] - 1.0) if nav.size else 0.0,
        annualised_return=metrics.annualised_return(nav, resolved_periods_per_year),
        annualised_vol=metrics.annualised_vol(rets, resolved_periods_per_year),
        sharpe=metrics.sharpe(
            rets,
            risk_free_rate / resolved_periods_per_year,
            resolved_periods_per_year,
        ),
        sortino=metrics.sortino(
            rets,
            risk_free_rate / resolved_periods_per_year,
            resolved_periods_per_year,
        ),
        max_drawdown=metrics.max_drawdown(nav),
        calmar=metrics.calmar(nav, resolved_periods_per_year),
        cvar=float(metrics.cvar(rets, prob=None, alpha=0.05, distribution_type="pnl"))
        if rets.size >= 2
        else 0.0,
        hit_rate=float(np.mean(rets > 0)) if rets.size else 0.0,
        avg_turnover=avg_turnover,
        total_cost=total_cost,
        n_periods=n_periods if n_periods is not None else int(max(nav.size - 1, 0)),
        n_warnings=n_warnings,
        n_solver_failures=n_solver_failures,
        held_fraction=held_fraction,
    )
