from dataclasses import dataclass


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

    decay: in MPO, multiply forecast by decay^step.
           1.0 = slow signal (trust far future),
           0.0 = fast signal (only trust today).
    """

    decay: float = 1.0


@dataclass(frozen=True)
class TransactionCost:
    """
    Cost of trading.

    cost: linear cost coefficient (e.g. half bid-ask spread).
       Applied as a * |z|.
    market_impact: market impact coefficient (typically ~1).
       Applied as b * sigma * |z|^exponent / volume^(exponent-1).
    exponent: power of the market impact term. Must be >= 1.
              1.0 = linear, 1.5 = square-root (Almgren), 2.0 = quadratic.
    """

    cost: float = 0.0
    market_impact: float = 1.0
    exponent: float = 1.5


@dataclass(frozen=True)
class HoldingCost:
    """
    Cost of holding positions overnight.
    short_fees: annualised borrowing fee in percent (e.g. 5.0 = 5% per year).
                Applied as short_fees_per_period * (w_plus)_negative.
    """

    short_fees: float = 5.0


@dataclass(frozen=True)
class WeightedTerm:
    weight: float
    spec: (
        ExpectedReturn
        | CovarianceRisk
        | TransactionCost
        | HoldingCost
        | CVaRRisk
        | CVaRCuttingPlane
    )


@dataclass(frozen=True)
class ObjectiveSpec:
    terms: tuple[WeightedTerm, ...]
