from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from qraft.construction.frontier.config import FrontierKind, RiskMeasure
from qraft.construction.optimization.optimization import SolverStatus


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    gamma: float
    status: SolverStatus
    metric_kind: FrontierKind
    objective_value: float | None = None
    expected_return: float | None = None
    volatility: float | None = None
    cvar_in_model: float | None = None
    cvar_terminal: float | None = None
    turnover: float | None = None
    cash_weight: float | None = None
    target_weights: dict[str, float] | None = None
    failure_message: str | None = None

    @classmethod
    def failed(
        cls,
        gamma: float,
        status: SolverStatus,
        message: str = "",
        *,
        metric_kind: FrontierKind = FrontierKind.EX_ANTE_MPO,
    ) -> "FrontierPoint":
        return cls(
            gamma=gamma,
            status=status,
            metric_kind=metric_kind,
            failure_message=message,
        )


@dataclass(frozen=True, slots=True)
class FrontierResult:
    risk_measure: RiskMeasure
    points: list[FrontierPoint]
    kind: FrontierKind
    assets: list[str]
    n_horizons: int
    n_scenarios: int
    dropped_assets: list[str]
    dropped_weight: float = 0.0

    def plot(
        self,
        *,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (8, 5),
    ) -> None:
        valid = [
            p
            for p in self.points
            if p.expected_return is not None
            and getattr(p, self.risk_measure) is not None
        ]
        if not valid:
            raise ValueError(
                f"No valid points to plot (risk_measure={self.risk_measure!r})."
            )

        risk = np.array([getattr(p, self.risk_measure) for p in valid])
        ret = np.array([p.expected_return for p in valid])
        gamma = np.array([p.gamma for p in valid])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        scatter = ax.scatter(
            risk, ret, c=gamma, cmap="viridis_r", s=28, zorder=3, edgecolors="none"
        )

        order = np.argsort(risk)
        ax.plot(risk[order], ret[order], linestyle="--", linewidth=1.0, alpha=0.5)

        ax.set_xlabel(self.risk_measure.replace("_", " ").title())
        ax.set_ylabel("Expected Return")
        ax.set_title("Efficient Frontier")
        ax.grid(True, alpha=0.3)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Risk aversion (γ)")

        plt.show()
