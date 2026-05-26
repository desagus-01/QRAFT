from typing import Literal, Protocol, cast, runtime_checkable

import cvxpy as cp
import numpy as np
from cvxpy import Constraint, Expression
from numpy.typing import NDArray

ConstraintType = Literal["hard", "soft"]


@runtime_checkable
class PortfolioConstraint(Protocol):
    """Any object with this method is a valid constraint."""

    @property
    def constraint_type(self) -> ConstraintType: ...

    @property
    def soft_weight(self) -> float: ...

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]: ...

    def violation_expr(self, weights: Expression, trades: Expression) -> Expression: ...


class LongOnly:
    """Require all post-trade weights to be non-negative."""

    def __init__(
        self, constraint_type: ConstraintType = "hard", soft_weight: float = 1.0
    ):
        self.constraint_type = constraint_type
        self.soft_weight = soft_weight

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [weights >= 0]

    def violation_expr(self, weights: Expression, trades: Expression) -> Expression:
        # pos(-w_i) = max(-w_i, 0): positive only when w_i < 0 (short).
        return cp.pos(-weights)


class FullyInvested:
    """Require weights to sum to 1 (no cash leakage)."""

    def __init__(
        self, constraint_type: ConstraintType = "hard", soft_weight: float = 1.0
    ):
        self.constraint_type = constraint_type
        self.soft_weight = soft_weight

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return cast(list[Constraint], [cp.sum(weights) == 1])

    def violation_expr(self, weights: Expression, trades: Expression) -> Expression:
        # Equality constraint: violation = |sum(w) - 1|.
        return cp.abs(cp.sum(weights) - 1)


class MaxWeight:
    """Cap each asset weight at ``limit``.

    Parameters
    ----------
    limit:
        Either a scalar applied to all assets, or a per-asset array of
        shape ``(n_assets,)``.
    """

    def __init__(
        self,
        limit: float | NDArray[np.floating],
        constraint_type: ConstraintType = "hard",
        soft_weight: float = 1.0,
    ):
        self.limit = limit
        self.constraint_type = constraint_type
        self.soft_weight = soft_weight

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [weights <= self.limit]

    def violation_expr(self, weights: Expression, trades: Expression) -> Expression:
        # pos(w_i - limit): positive only when w_i exceeds the cap.
        return cp.pos(weights - self.limit)


class MaxWeightTopN:
    """Cap the sum of the largest ``top_n`` weights at ``sum_limit``.

    Parameters
    ----------
    top_n:
        Number of largest weights to include in the cap.
    sum_limit:
        Maximum allowed sum of the top ``top_n`` weights.
    """

    def __init__(
        self,
        top_n: int,
        sum_limit: float,
        constraint_type: ConstraintType = "hard",
        soft_weight: float = 1.0,
    ):
        if top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        self.top_n = top_n
        self.sum_limit = sum_limit
        self.constraint_type = constraint_type
        self.soft_weight = soft_weight

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [cp.sum_largest(weights, self.top_n) <= self.sum_limit]

    def violation_expr(self, weights: Expression, trades: Expression) -> Expression:
        # Scalar violation: positive only when the top-N sum exceeds the cap.
        return cp.pos(cp.sum_largest(weights, self.top_n) - self.sum_limit)


class MinWeight:
    """Require each asset weight to be at least ``limit``.

    Parameters
    ----------
    limit:
        Either a scalar applied to all assets, or a per-asset array of
        shape ``(n_assets,)``.
    """

    def __init__(
        self,
        limit: float | NDArray[np.floating],
        constraint_type: ConstraintType = "hard",
        soft_weight: float = 1.0,
    ):
        self.limit = limit
        self.constraint_type = constraint_type
        self.soft_weight = soft_weight

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [weights >= self.limit]

    def violation_expr(self, weights: Expression, trades: Expression) -> Expression:
        # pos(limit - w_i): positive only when w_i falls below the floor.
        return cp.pos(self.limit - weights)


class TurnoverLimit:
    """Limit the L1 norm of trades (one-way turnover) per period.

    Parameters
    ----------
    limit:
        Maximum allowed ``||trades||_1`` per horizon step.
    """

    def __init__(
        self,
        limit: float | NDArray[np.floating],
        constraint_type: ConstraintType = "hard",
        soft_weight: float = 1.0,
    ):
        self.limit = limit
        self.constraint_type = constraint_type
        self.soft_weight = soft_weight

    def compile_to_cvxpy(
        self, weights: Expression, trades: Expression
    ) -> list[Constraint]:
        return [0.5 * cp.norm1(trades) <= self.limit]

    def violation_expr(self, weights: Expression, trades: Expression) -> Expression:
        # Scalar: pos(||z||_1 - limit).
        return cp.pos(cp.norm1(trades) - self.limit)
