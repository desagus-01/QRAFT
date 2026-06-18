from dataclasses import dataclass

from qraft.construction.policies.policies import PolicyProtocol
from qraft.construction.policies.policy_decision import PolicyDecision
from qraft.construction.policies.policy_projection import PolicyProjection
from qraft.construction.state import PortfolioState
from qraft.forecast.forecast_paths import ForecastPaths


@dataclass(frozen=True, slots=True)
class PolicyRun:
    """The decision and its projection from one policy invocation."""

    decision: PolicyDecision
    projection: PolicyProjection


def run_policy(
    policy: PolicyProtocol,
    state: PortfolioState,
    forecasts: ForecastPaths,
) -> PolicyRun:
    """Make a policy decision and project it through the supplied forecasts."""

    decision = policy.decide(state, forecasts)
    projection = PolicyProjection.from_decision(decision, forecasts, state)
    return PolicyRun(decision=decision, projection=projection)
