from dataclasses import dataclass
from typing import Any

from qraft.construction.optimization.inputs import PolicyInputs
from qraft.construction.policies.policies import PolicyProtocol
from qraft.construction.policies.policy_decision import PolicyDecision
from qraft.construction.policies.policy_projection import PolicyProjection
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import ForecastPaths


@dataclass(frozen=True, slots=True)
class PolicyRun:
    """The decision and its projection from one policy invocation."""

    decision: PolicyDecision
    projection: PolicyProjection | None = None


def run_policy(
    policy: PolicyProtocol,
    state: PortfolioState,
    forecasts: ForecastPaths | None = None,
    policy_inputs: PolicyInputs | None = None,
    inputs: dict[str, Any] | None = None,
) -> PolicyRun:
    """Make a policy decision and optionally project it through supplied forecasts."""

    decision = policy.decide(state, policy_inputs, inputs=inputs)

    projection = (
        PolicyProjection.from_decision(decision, forecasts, state)
        if forecasts is not None
        else None
    )
    return PolicyRun(decision=decision, projection=projection)
