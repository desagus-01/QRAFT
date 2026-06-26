from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product
from typing import Any

from qraft.backtest.selection.overlays import apply_problem_hyperparameters
from qraft.backtest.selection.results import PolicyCandidate, PolicyParams
from qraft.construction.policies import MPOPolicy, PolicyProtocol


def apply_hyperparameters(
    policy: PolicyProtocol,
    params: PolicyParams | Mapping[str, Any],
) -> PolicyProtocol:
    """Return an immutable policy variant with candidate params applied."""
    normalized = (
        params if isinstance(params, PolicyParams) else PolicyParams.of(**params)
    )
    if isinstance(policy, MPOPolicy):
        problem = apply_problem_hyperparameters(policy.problem, normalized)
        name = policy.name if not normalized else f"{policy.name}[{normalized}]"
        return replace(policy, problem=problem, name=name, _optimizer_cache={})

    if normalized:
        raise TypeError(
            f"{type(policy).__name__} does not expose tunable hyperparameters"
        )
    return policy


def param_grid(grid: Mapping[str, Sequence[Any]]) -> tuple[PolicyParams, ...]:
    """Return deterministic Cartesian-product parameter sets from a value grid."""
    if not grid:
        return (PolicyParams.of(),)

    keys = tuple(grid)
    params: list[PolicyParams] = []

    for values in product(*(grid[key] for key in keys)):
        pairs = tuple(sorted(zip(keys, values, strict=True)))
        params.append(PolicyParams(pairs))

    return tuple(params)


def expand_candidates(
    policy: PolicyProtocol,
    grid: Mapping[str, Sequence[Any]],
) -> tuple[PolicyCandidate, ...]:
    """Expand a base policy and parameter grid into policy candidates."""
    return tuple(
        PolicyCandidate(
            params=params,
            policy=apply_hyperparameters(policy, params),
        )
        for params in param_grid(grid)
    )
