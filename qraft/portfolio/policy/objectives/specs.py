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

    cost:
        Linear cost coefficient (e.g. half bid-ask spread).
        Applied as ``cost * |z_i|`` uniformly across assets.

    pershare_cost:
        Flat dollar cost per share traded (e.g. $0.005/share for US equities).
        Converted to weight-space as ``pershare_cost / price_i``, so cheap
        stocks are penalised more heavily than expensive ones.
        Requires ``inputs["prices"]`` (shape ``(n_assets,)``) at solve time.
        Set to 0.0 (default) to disable.

    market_impact:
        Market-impact coefficient (typically ~1).
        Applied as ``market_impact * sigma_i * |z_i|^exponent / volume_i^(exponent-1)``.

    exponent:
        Power of the market-impact term. Must be >= 1.
        1.0 = linear, 1.5 = square-root (Almgren), 2.0 = quadratic.

    c_bias:
        Directional linear slippage coefficient, applied as ``c_bias * z_i``
        (note: signed, not absolute).  Use this to model execution price
        differing from the open:
          - Set to the *negative* of the expected open-to-VWAP return if you
            execute at VWAP (you systematically pay more when buying).
          - Set to the *negative* of the open-to-close return if you execute
            at close.
        A positive ``c_bias`` penalises buying and rewards selling;
        a negative value does the opposite.  Default 0.0 (disabled).
    """

    cost: float = 0.0
    pershare_cost: float = 0.0
    market_impact: float = 1.0
    exponent: float = 1.5
    c_bias: float = 0.0


@dataclass(frozen=True)
class HoldingCost:
    """
    Cost of holding positions overnight.

    All fee parameters are **annualised percentages** (e.g. 5.0 = 5% p.a.).
    They are converted to per-period rates internally using ``periods_per_year``
    via the linear approximation ``fee_per_period = annual_pct / (100 * ppy)``.

    short_fees:
        Borrowing fee on *short* positions (negative weights).
        Applied as ``short_fees_per_period * sum(neg(w))``.
        Default 5.0 (5% p.a.).

    long_fees:
        Fee on *long* positions — custody/financing charges (e.g. prime
        brokerage long financing, ETF management fees).
        Applied as ``long_fees_per_period * sum(pos(w))``.
        Default 0.0 (disabled). Typical range: 0.1–0.5% p.a.

    dividends:
        Dividend yield per period — only relevant when your return series
        are **price returns** (not total returns / adjusted prices).
        If you use total returns (the default for most data providers),
        dividends are already embedded in returns; leave this at 0.0.
        Applied as ``+dividends_per_period * sum(w)`` (income, positive
        contribution to the objective).
        Default 0.0 (disabled).

    periods_per_year:
        Number of trading periods in a calendar year used to scale annual
        fees down to per-period rates.
        252 for daily, 52 for weekly, 12 for monthly.  Default 252.
    """

    short_fees: float = 5.0
    long_fees: float = 0.0
    dividends: float = 0.0
    periods_per_year: int = 252


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
