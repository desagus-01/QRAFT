from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, runtime_checkable

from qraft.construction.optimization.moments import (
    InputPlan,
    PolicyInputs,
    RequiredPolicyInputs,
)
from qraft.core.snapshot import (
    MarketSnapshot,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class PolicyInputsProvider(Protocol):
    """Maps a decision snapshot to the PolicyInputs the policy optimises against."""

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs: ...


@runtime_checkable
class PolicyInputRequirements(Protocol):
    def required_inputs(self) -> RequiredPolicyInputs: ...


def policy_risk_source(
    input_config: InputPlan,
    policy: PolicyInputRequirements | None = None,
):
    if input_config.risk is not None:
        validate_policy_risk_source(input_config.risk, policy)
        return input_config.risk
    if policy is None:
        return "both"
    return policy.required_inputs().risk_source


def validate_policy_risk_source(risk, policy: PolicyInputRequirements | None) -> None:
    if policy is None:
        return
    required = policy.required_inputs()
    has_covariance = risk in {"covariance", "both"}
    has_scenarios = risk in {"cvar", "both"}
    missing: list[str] = []
    if required.covariances and not has_covariance:
        missing.append("covariances")
    if required.scenarios and not has_scenarios:
        missing.append("scenario_returns")
    if missing:
        raise ValueError(
            f"input_config.risk={risk!r} does not satisfy policy requirements: "
            f"missing {', '.join(missing)}. Omit risk to infer it from the "
            "policy, or use risk='both'."
        )


@dataclass
class DateCache:
    inner: PolicyInputsProvider
    _table: dict[datetime, PolicyInputs] = field(default_factory=dict, init=False)
    _universe: tuple[str, ...] | None = field(default=None, init=False)

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
        universe = tuple(snapshot.universe.all_tickers)
        if self._universe is None:
            self._universe = universe
        elif universe != self._universe:
            raise ValueError(
                "DateCache requires a fixed universe; the asset set changed "
                "mid-run. Use a fresh provider per universe."
            )
        cached = self._table.get(snapshot.t)
        if cached is None:
            cached = self.inner.for_date(snapshot, step)
            self._table[snapshot.t] = cached
        return cached


@dataclass(frozen=True, slots=True)
class PrecomputedInputsProvider:
    """Serve PolicyInputs from a ``{date: PolicyInputs}`` table (BYO moments)."""

    table: Mapping[datetime, PolicyInputs]

    def for_date(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
        inputs = self.table.get(snapshot.t)
        if inputs is None:
            raise KeyError(f"No precomputed PolicyInputs for {snapshot.t!r}")
        return inputs
