from dataclasses import dataclass
from typing import Any

from qraft.construction.optimization.inputs import PolicyInputs
from qraft.construction.policies.policies import PolicyProtocol
from qraft.construction.policies.policy_decision import PolicyDecision
from qraft.construction.policies.policy_projection import PolicyProjection
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import ForecastPaths
from qraft.risk.feature_selection import Criterion
from qraft.risk.risk_report import PortfolioRisk


@dataclass(frozen=True, slots=True)
class PolicyRun:
    """The decision and its projection from one policy invocation."""

    decision: PolicyDecision
    projection: PolicyProjection
    forecasts: ForecastPaths | None = None

    def risk(
        self,
        *,
        auto_select_factors: bool = False,
        criterion: Criterion | None = None,
    ) -> PortfolioRisk:
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
    policy_inputs: PolicyInputs | None = None,
    inputs: dict[str, Any] | None = None,
) -> PolicyRun:
    """Make a policy decision and optionally project it through supplied forecasts."""

    decision = policy.decide(state, policy_inputs, inputs=inputs)

    projection = PolicyProjection.from_decision(decision, forecasts, state)

    return PolicyRun(decision=decision, projection=projection, forecasts=forecasts)
