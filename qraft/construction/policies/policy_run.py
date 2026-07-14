"""Policy-run record and execution helper."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt

from qraft.construction.optimization.diagnostics import (
    forecast_plan_metrics,
    forecast_terminal_cvar,
    in_model_cvar,
)
from qraft.construction.optimization.inputs import OptimizerInputs
from qraft.construction.optimization.optimization import MPOResult
from qraft.construction.policies.policies import PolicyProtocol
from qraft.construction.policies.policy_decision import PolicyDecision
from qraft.construction.policies.policy_projection import PolicyProjection
from qraft.construction.state import PortfolioState
from qraft.core.scenarios.view_types import ViewDiagnostics
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.risk.feature_selection import Criterion
from qraft.risk.risk_report import PortfolioRisk


@dataclass(frozen=True, slots=True)
class PolicyRun:
    """The decision and its projection from one policy invocation."""

    decision: PolicyDecision
    projection: PolicyProjection
    forecasts: ForecastPaths | None = None
    optimizer_inputs: OptimizerInputs | None = None
    as_of: datetime | None = None

    @property
    def target_weights(self) -> dict[str, float]:
        """Return risky target weights keyed by asset."""
        return dict(
            zip(
                self.decision.asset_order,
                self.decision.target_weights_risk.tolist(),
            )
        )

    @property
    def view_diagnostics(self) -> ViewDiagnostics | None:
        """Return entropy-pooling diagnostics carried by optimizer inputs, if any."""
        return getattr(self.optimizer_inputs, "view_diagnostics", None)

    def mpo_result(self) -> MPOResult:
        """Return optimizer diagnostics for policies backed by an MPO solve."""
        diagnostics = self.decision.diagnostics
        if not isinstance(diagnostics, MPOResult):
            raise ValueError(
                "PolicyRun diagnostics require MPOResult diagnostics; this policy "
                "did not produce optimizer diagnostics."
            )
        if self.optimizer_inputs is None:
            raise ValueError(
                "PolicyRun diagnostics require optimizer_inputs. Pass optimizer_inputs to "
                "run_policy() or use Allocation.at() with an input plan."
            )
        return diagnostics

    def plan_metrics(self) -> dict[str, float]:
        """Return diagnostic metrics for the optimizer's forecast plan."""
        return forecast_plan_metrics(self.mpo_result(), self.optimizer_inputs)

    def terminal_cvar(self, alpha: float = 0.05) -> float:
        """Return terminal forecast CVaR for the optimized plan."""
        return forecast_terminal_cvar(self.mpo_result(), self.optimizer_inputs, alpha)

    def in_model_cvar(self, alpha: float = 0.05) -> float:
        """Return in-model CVaR from the optimizer inputs and solution."""
        return in_model_cvar(self.mpo_result(), self.optimizer_inputs, alpha)

    def plot_weights(self, top_n: int | None = None):
        """Plot target risky weights plus cash weight."""
        weights = {**self.target_weights, "cash": self.decision.target_cash_weight}
        items = sorted(weights.items(), key=lambda item: item[1])
        if top_n is not None:
            items = sorted(items, key=lambda item: abs(item[1]), reverse=True)[:top_n]
            items = sorted(items, key=lambda item: item[1])

        labels = [label for label, _ in items]
        values = [value for _, value in items]
        fig, ax = plt.subplots()
        ax.barh(labels, values)
        ax.set_xlabel("Weight")
        ax.set_title("Target weights")
        fig.tight_layout()
        return fig

    def risk(
        self,
        *,
        auto_select_factors: bool = False,
        criterion: Criterion | None = None,
    ) -> PortfolioRisk:
        """Compute risk from this run's existing projection and forecasts."""
        if self.projection is None or self.forecasts is None:
            raise ValueError("PolicyRun.risk requires both projection and forecasts")
        return self.projection.risk(
            self.forecasts,
            auto_select_factors=auto_select_factors,
            criterion=criterion,
        )


def run_policy(
    policy: PolicyProtocol,
    state: PortfolioState,
    forecasts: ForecastPaths,
    optimizer_inputs: OptimizerInputs | None = None,
    inputs: dict[str, Any] | None = None,
    as_of: datetime | None = None,
) -> PolicyRun:
    """Make a policy decision and optionally project it through supplied forecasts."""
    decision = policy.decide(state, optimizer_inputs, inputs=inputs)

    projection = PolicyProjection.from_decision(decision, forecasts, state)

    return PolicyRun(
        decision=decision,
        projection=projection,
        forecasts=forecasts,
        optimizer_inputs=optimizer_inputs,
        as_of=as_of,
    )
