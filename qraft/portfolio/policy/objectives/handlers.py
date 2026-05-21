from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from portfolio.policy.objectives.protocol import register_objective
from portfolio.policy.objectives.specs import (
    CovarianceRisk,
    CVaRRisk,
    ExpectedReturn,
    HoldingCost,
    TransactionCost,
)


def _project_on_psd_cone_and_factorize(
    covariance: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Return F such that F @ F.T ≈ covariance, with negative eigenvalues clamped.

    DPP-compliant CVXPY expression is ``cp.sum_squares(F.T @ w)``.
    """
    cov = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals))


@register_objective(ExpectedReturn)
class ExpectedReturnHandler:
    """
    Maximise the probability-weighted expected return at each horizon.

    The optional ``decay`` on the spec discounts the forecast confidence for
    far-future horizons:  contribution at step h = decay^h * mean_h @ w_h.
    """

    def allocate(
        self, spec: ExpectedReturn, horizons: int, n_assets: int, **_kwargs
    ) -> dict[str, Any]:
        return {"mean": cp.Parameter((horizons, n_assets), name="mean")}

    def compile(
        self,
        spec: ExpectedReturn,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        decay_factor = spec.decay**horizon
        return (decay_factor * (params["mean"][horizon, :] @ weights_h), [])

    def update(
        self, spec: ExpectedReturn, params: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """
        Expected ``inputs`` keys:
          ``"moments"``  – a ``HorizonMoments`` instance.
        """
        params["mean"].value = inputs["moments"].mean


@register_objective(CVaRRisk)
class CVaRRiskHandler:
    """
    Penalise Conditional Value-at-Risk via the Rockafellar-Uryasev LP.

    For each horizon h with scenario returns R_h (shape (S, N)),
    weight w_h, and probabilities p (shape (S,)):

        CVaR_α(w_h) = min over ζ, u≥0 of   ζ + (1/α) · p^T u
                     s.t.                  u ≥ -R_h w_h - ζ

    Returned as a maximisation contribution: -CVaR.
    """

    def allocate(
        self, spec: CVaRRisk, horizons: int, n_assets: int, n_scenarios: int
    ) -> dict[str, cp.Parameter | cp.Variable]:
        if not 0.0 < spec.alpha <= 1.0:
            raise ValueError(f"CVaR alpha must be in (0, 1], got {spec.alpha}")

        params: dict[str, Any] = {
            "probs": cp.Parameter(n_scenarios, name="cvar_probs", nonneg=True),
        }
        for h in range(horizons):
            params[f"R_{h}"] = cp.Parameter((n_scenarios, n_assets), name=f"cvar_R_{h}")
            params[f"zeta_{h}"] = cp.Variable(name=f"cvar_zeta_{h}")
            params[f"u_{h}"] = cp.Variable(n_scenarios, nonneg=True, name=f"cvar_u_{h}")
        return params

    def compile(
        self,
        spec: CVaRRisk,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        R = params[f"R_{horizon}"]
        p = params["probs"]
        zeta = params[f"zeta_{horizon}"]
        u = params[f"u_{horizon}"]

        loss = -R @ weights_h
        cvar = zeta + (1.0 / spec.alpha) * (p @ u)

        aux = [u >= loss - zeta]
        return -cvar, aux

    def update(
        self,
        spec: CVaRRisk,
        params: dict[str, Any],
        inputs: dict[str, Any],
    ) -> None:
        moments = inputs["moments"]
        params["probs"].value = np.asarray(moments.scenario_probs, dtype=float)
        for h in range(moments.n_horizons):
            key = f"R_{h}"
            if key in params:
                params[key].value = moments.scenario_returns[:, h, :]


@register_objective(CovarianceRisk)
class CovarianceRiskHandler:
    """
    Penalise quadratic portfolio variance at each horizon.
    """

    def allocate(
        self, spec: CovarianceRisk, horizons: int, n_assets: int, **_kwargs
    ) -> dict[str, cp.Parameter]:
        return {
            f"cov_sqrt_{h}": cp.Parameter(
                (n_assets, n_assets),
                name=f"cov_sqrt_{h}",
            )
            for h in range(horizons)
        }

    def compile(
        self,
        spec: CovarianceRisk,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        # sum_squares(affine_in_w) is always convex → DCP + DPP compliant.
        return (-cp.sum_squares(params[f"cov_sqrt_{horizon}"].T @ weights_h), [])

    def update(
        self, spec: CovarianceRisk, params: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """
        Expected ``inputs`` keys:
          ``"moments"``  – a ``HorizonMoments`` instance.
        """
        moments = inputs["moments"]
        for h in range(moments.n_horizons):
            key = f"cov_sqrt_{h}"
            if key in params:
                params[key].value = _project_on_psd_cone_and_factorize(
                    moments.covariances[h]
                )


@register_objective(TransactionCost)
class TransactionCostHandler:
    """
    Penalise trading costs at each horizon.

    Two components are combined:

    1. **Linear cost** (e.g. half bid-ask spread)::

           a_i * |z_i|   where  a_i = spec.cost  (uniform across assets)

    2. **Market-impact cost** (Almgren-style power law)::

           b * sigma_i / V_i^(p-1) * |z_i|^p

       where ``b = spec.market_impact``, ``sigma_i`` is the per-asset
       period volatility (derived from the horizon-0 covariance diagonal),
       ``V_i`` is the per-asset average daily volume (from ``inputs``), and
       ``p = spec.exponent`` (1.5 by default → square-root impact).

    If volume data is unavailable, the impact coefficient falls back to
    ``b * sigma_i`` (i.e. the volume denominator is dropped).

    The expression is negative so that maximising it minimises costs.
    """

    def allocate(
        self, spec: TransactionCost, horizons: int, n_assets: int, **_kwargs
    ) -> dict[str, Any]:
        return {
            # Per-asset linear cost coefficient  a_i
            "tc_linear": cp.Parameter(n_assets, nonneg=True, name="tc_linear"),
            # Per-asset market-impact coefficient  b * sigma_i / V_i^(p-1)
            "tc_impact": cp.Parameter(n_assets, nonneg=True, name="tc_impact"),
        }

    def compile(
        self,
        spec: TransactionCost,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        # Linear term: sum_i a_i * |z_i|
        linear = cp.sum(cp.multiply(params["tc_linear"], cp.abs(trades_h)))

        # Power-law impact term: sum_i c_i * |z_i|^p
        # cp.power(cp.abs(z), p) is valid for p >= 1 via CVXPY's power-cone
        # support; nonneg parameters scaling a convex expression remain DPP.
        impact = cp.sum(
            cp.multiply(
                params["tc_impact"],
                cp.power(cp.abs(trades_h), spec.exponent),
            )
        )

        return (-1.0 * (linear + impact), [])  # type: ignore[return-value]  # cvxpy stubs under-specify cp.power return type

    def update(
        self, spec: TransactionCost, params: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """
        Expected ``inputs`` keys:
          ``"moments"``  – a ``HorizonMoments`` instance.
          ``"volume"``   – per-asset ADV, shape ``(n_assets,)`` (optional).
        """
        moments = inputs["moments"]
        n = moments.n_assets

        # Uniform linear cost
        params["tc_linear"].value = np.full(n, spec.cost)

        # Per-asset volatility from the horizon-0 covariance diagonal
        sigma = np.sqrt(np.maximum(np.diag(moments.covariances[0]), 0.0))

        volume: NDArray[np.floating] | None = inputs.get("volume")
        if volume is not None and np.all(volume > 0):
            vol_factor = volume ** (spec.exponent - 1.0)
            impact_coeff = spec.market_impact * sigma / vol_factor
        else:
            # Degrade gracefully: drop the volume denominator
            impact_coeff = spec.market_impact * sigma

        params["tc_impact"].value = np.maximum(impact_coeff, 0.0)


@register_objective(HoldingCost)
class HoldingCostHandler:
    """
    Penalise the cost of holding short positions overnight.

    The per-period short fee is::

        short_fees_per_period = (spec.short_fees / 100) / periods_per_year

    Applied as::

        -short_fees_per_period * sum_i max(-w_i, 0)

    ``cp.neg(x) = max(-x, 0)``, so ``cp.sum(cp.neg(w))`` is the total
    magnitude of short weight, charged at the borrowing rate each period.

    The expression is negative so that maximising it minimises holding costs.
    """

    def allocate(
        self, spec: HoldingCost, horizons: int, n_assets: int, **_kwargs
    ) -> dict[str, Any]:
        return {"hc_short_rate": cp.Parameter(nonneg=True, name="hc_short_rate")}

    def compile(
        self,
        spec: HoldingCost,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        return (-params["hc_short_rate"] * cp.sum(cp.neg(weights_h)), [])

    def update(
        self, spec: HoldingCost, params: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """
        Expected ``inputs`` keys:
          ``"periods_per_year"``  – int, trading periods per year (default 252).
        """
        periods_per_year: int = inputs.get("periods_per_year", 252)
        annual_rate = spec.short_fees / 100.0
        params["hc_short_rate"].value = annual_rate / periods_per_year
