from dataclasses import dataclass

from numpy import floating
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CovarianceRisk:
    """
    Penalize quadratic risk: weight.T @ covariance @ weight.
    """

    pass


@dataclass(frozen=True, slots=True)
class CVaRRisk:
    """Rockafellar-Uryasev LP formulation of CVaR."""

    alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class CVaRCuttingPlane:
    """Cutting-plane approximation of CVaR (supports iterative refinement)."""

    alpha: float = 0.05
    max_cuts: int = 200
    tol: float = 1e-6


@dataclass(frozen=True)
class ExpectedReturn:
    """
    Reward expected return: mean @ weight
    """

    decay: float = 1.0


@dataclass(frozen=True)
class CashReturn:
    """
    Reward the return earned on the explicit cash position.
    """

    pass


@dataclass(frozen=True)
class TransactionCost:
    """
    Cost of trading.
    """

    cost: float = 0.0
    pershare_cost: float = 0.0
    market_impact: float = 1.0
    exponent: float = 1.5
    c_bias: float | NDArray[floating] = 0.0


@dataclass(frozen=True)
class HoldingCost:
    """
    Cost of holding positions overnight.
    """

    short_fees: float
    long_fees: float
    dividends: float
    periods_per_year: int = 252


@dataclass(frozen=True)
class WeightedTerm:
    weight: float
    spec: (
        ExpectedReturn
        | CashReturn
        | CovarianceRisk
        | TransactionCost
        | HoldingCost
        | CVaRRisk
        | CVaRCuttingPlane
    )


@dataclass(frozen=True)
class ObjectiveSpec:
    terms: tuple[WeightedTerm, ...]
