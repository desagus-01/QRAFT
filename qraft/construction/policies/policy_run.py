from dataclasses import dataclass
from typing import Any

from qraft.construction.market_snapshot import MarketSnapshot
from qraft.construction.optimization.moments import PolicyInputs
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
    snapshot: MarketSnapshot,
    state: PortfolioState,
    forecasts: ForecastPaths | None = None,
    policy_inputs: PolicyInputs | None = None,
    inputs: dict[str, Any] | None = None,
) -> PolicyRun:
    """Make a policy decision and optionally project it through supplied forecasts."""

    if policy_inputs is not None and hasattr(policy, "make_decision"):
        decision = policy.make_decision(
            snapshot,
            state,
            policy_inputs=policy_inputs,
            inputs=inputs,
        )
    else:
        decision = policy.decide(snapshot, state)

    projection = (
        PolicyProjection.from_decision(decision, forecasts, state)
        if forecasts is not None
        else None
    )
    return PolicyRun(decision=decision, projection=projection)
