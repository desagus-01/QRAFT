from typing import Protocol, cast, runtime_checkable

import cvxpy as cp
import numpy as np
from cvxpy import Constraint, Expression
from numpy.typing import NDArray


@runtime_checkable
class PortfolioConstraint(Protocol):
    """Any object with this method is a valid constraint."""

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]: ...


class LongOnly:
    """Require all post-trade weights to be non-negative."""

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [weights >= 0]


class FullyInvested:
    """Require weights to sum to 1 (no cash leakage)."""

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return cast(list[Constraint], [cp.sum(weights) == 1])


class MaxWeight:
    """Cap each asset weight at ``limit``.

    Parameters
    ----------
    limit:
        Either a scalar applied to all assets, or a per-asset array of
        shape ``(n_assets,)``.
    """

    def __init__(self, limit: float | NDArray[np.floating]):
        self.limit = limit

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [weights <= self.limit]


class MinWeight:
    """Require each asset weight to be at least ``limit``.

    Parameters
    ----------
    limit:
        Either a scalar applied to all assets, or a per-asset array of
        shape ``(n_assets,)``.
    """

    def __init__(self, limit: float | NDArray[np.floating]):
        self.limit = limit

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [weights >= self.limit]


class TurnoverLimit:
    """Limit the L1 norm of trades (one-way turnover) per period.

    Parameters
    ----------
    limit:
        Maximum allowed ``||trades||_1`` per horizon step.
    """

    def __init__(self, limit: float):
        self.limit = limit

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [cp.norm1(trades) <= self.limit]


class FactorExposureLimit:
    """Cap the signed portfolio exposure to a factor: ``factor @ weights <= limit``.

    Parameters
    ----------
    factor:
        Factor loadings vector of shape ``(n_assets,)``.
    limit:
        Maximum allowed exposure.
    """

    def __init__(self, factor: NDArray[np.floating], limit: float):
        self.factor = factor
        self.limit = limit

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [self.factor @ weights <= self.limit]


DEFAULT_CONSTRAINTS: list[PortfolioConstraint] = [LongOnly(), FullyInvested()]
