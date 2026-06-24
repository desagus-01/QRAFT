from dataclasses import dataclass

import numpy as np
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
    max_cuts: int = (
        30  # TODO: Make this dynamic as currently is what takes the most gbs
    )
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

    cost: float = 0.0005
    pershare_cost: float = 0.0
    market_impact: float = 0.0
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


def transaction_cost_coeffs(
    spec: TransactionCost,
    *,
    n_assets: int,
    prices: NDArray[floating] | None = None,
    sigma: NDArray[floating] | None = None,
    volume: NDArray[floating] | None = None,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]:
    """Per-asset (linear, impact, bias) weight-space cost coefficients.

    Single source of truth for trading cost, shared by the optimizer penalty
    (``objectives.handlers.TransactionCostHandler``) and realised execution
    (``backtest.execution.execute_with_costs``). Mirrors the original handler
    coefficient logic so ex-ante and ex-post can never drift.
    """
    linear = np.full(n_assets, spec.cost, dtype=float)
    if spec.pershare_cost > 0.0 and prices is not None and np.all(prices > 0):
        linear = linear + spec.pershare_cost / np.asarray(prices, dtype=float)
    linear = np.maximum(linear, 0.0)

    if spec.market_impact == 0.0 or sigma is None:
        impact = np.zeros(n_assets, dtype=float)
    else:
        sig = np.asarray(sigma, dtype=float)
        if volume is not None and np.all(volume > 0):
            sig = sig / (np.asarray(volume, dtype=float) ** (spec.exponent - 1.0))
        impact = np.maximum(spec.market_impact * sig, 0.0)

    bias = np.broadcast_to(np.asarray(spec.c_bias, dtype=float), (n_assets,)).copy()
    return linear, impact, bias


def holding_cost_rates(
    spec: HoldingCost,
) -> tuple[float, float, float]:
    """Per-period holding cost rates (short_fees, long_fees, dividends) per bar."""
    ppy = spec.periods_per_year
    return spec.short_fees / ppy, spec.long_fees / ppy, spec.dividends / ppy


def holding_cost_value(
    spec: HoldingCost,
    weights: NDArray[floating],
    *,
    nav: float,
    n_periods: int = 1,
) -> float:
    """Realised holding cost in NAV units over ``n_periods`` bars.

    Sign convention: positive = cost to the portfolio (reduces NAV).
        cost = NAV · ( short_rate · sum(neg(w)) + long_rate · sum(pos(w))
                       - div_rate · sum(w) ) · n_periods
    """
    short_rate, long_rate, div_rate = holding_cost_rates(spec)
    short_cost = short_rate * float(np.sum(np.maximum(-weights, 0.0)))
    long_cost = long_rate * float(np.sum(np.maximum(weights, 0.0)))
    div_income = div_rate * float(np.sum(weights))
    return nav * (short_cost + long_cost - div_income) * n_periods


def transaction_cost_value(
    spec: TransactionCost,
    weight_trades: NDArray[floating],
    *,
    nav: float,
    prices: NDArray[floating] | None = None,
    sigma: NDArray[floating] | None = None,
    volume: NDArray[floating] | None = None,
) -> float:
    """Realised trading cost in NAV units for one rebalance.

    ``weight_trades`` are realised post-trade weight changes z_i = Δvalueᵢ/NAV.
    Equals the negated value of TransactionCostHandler's objective term:
        NAV · ( linear·|z| + impact·|z|^p + bias·z ).
    """
    z = np.asarray(weight_trades, dtype=float)
    linear, impact, bias = transaction_cost_coeffs(
        spec, n_assets=z.shape[0], prices=prices, sigma=sigma, volume=volume
    )
    abs_z = np.abs(z)
    per_weight = linear @ abs_z + impact @ (abs_z**spec.exponent) + bias @ z
    return float(nav * per_weight)


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
