from typing import Any

import cvxpy as cp
import numpy as np
from construction.optimization.moments import HorizonMoments
from construction.optimization.objectives.protocol import register_objective
from construction.optimization.objectives.specs import (
    CashReturn,
    CovarianceRisk,
    CVaRCuttingPlane,
    CVaRRisk,
    ExpectedReturn,
    HoldingCost,
    TransactionCost,
)
from numpy.typing import NDArray


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


def _cvar_alpha_check(alpha: float) -> None:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"CVaR alpha must be in (0, 1], got {alpha}")


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
        *,
        cash_h: cp.Expression | None = None,
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


@register_objective(CashReturn)
class CashReturnHandler:
    """
    Reward the return earned on the explicit scalar cash position.

    This handler ignores ``weights_h`` and ``trades_h``; it uses only the
    ``cash_h`` keyword argument supplied by :class:`MultiPeriodOptimizer`.
    """

    def allocate(
        self, spec: CashReturn, horizons: int, n_assets: int, **_kwargs
    ) -> dict[str, Any]:
        return {"cash_return": cp.Parameter(horizons, name="cash_return")}

    def compile(
        self,
        spec: CashReturn,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
        *,
        cash_h: cp.Expression | None = None,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        if cash_h is None:
            raise ValueError(
                "CashReturnHandler.compile requires cash_h to be provided. "
                "Ensure the optimizer has explicit cash enabled."
            )
        return (params["cash_return"][horizon] * cash_h, [])

    def update(
        self, spec: CashReturn, params: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """
        Populate the cash return parameter from ``inputs["moments"].cash_return``.

        Expected ``inputs`` keys:
          ``"moments"`` – a :class:`HorizonMoments` instance whose
          ``cash_return`` field is a 1-D array of shape ``(1,)`` or
          ``(horizons,)``.
        """
        moments = inputs["moments"]
        cash_ret = np.asarray(moments.cash_return, dtype=float).ravel()
        horizons = params["cash_return"].shape[0]

        if cash_ret.size == 1:
            # Single rate — broadcast to every planning horizon.
            cash_ret = np.full(horizons, cash_ret[0])
        elif cash_ret.size != horizons:
            raise ValueError(
                f"HorizonMoments.cash_return has {cash_ret.size} value(s) but "
                f"the optimizer was built with {horizons} horizon(s). "
                "Provide either a scalar or a vector of length horizons."
            )

        params["cash_return"].value = cash_ret


@register_objective(CVaRCuttingPlane)
class CVaRCuttingPlaneHandler:
    def allocate(
        self, spec: CVaRCuttingPlane, horizons: int, n_assets: int, n_scenarios: int
    ) -> dict[str, cp.Parameter | cp.Variable]:
        _cvar_alpha_check(spec.alpha)

        if spec.max_cuts < 3:
            raise ValueError(
                "CVaRCuttingPlane.max_cuts must be at least 3: "
                "2 anchor cuts plus at least 1 refinement cut."
            )

        params: dict[str, Any] = {
            "n_scenarios": n_scenarios,
            "cut_count": [0] * horizons,
        }

        for h in range(horizons):
            params[f"theta_{h}"] = cp.Variable(name=f"cvar_theta_{h}")
            params[f"zeta_{h}"] = cp.Variable(name=f"cvar_zeta_{h}")
            params[f"gw_{h}"] = cp.Parameter(
                (spec.max_cuts, n_assets), name=f"cvar_gw_{h}"
            )
            params[f"gz_{h}"] = cp.Parameter(spec.max_cuts, name=f"cvar_gz_{h}")
        return params

    def compile(
        self,
        spec: CVaRCuttingPlane,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
        *,
        cash_h: cp.Expression | None = None,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        theta = params[f"theta_{horizon}"]
        zeta = params[f"zeta_{horizon}"]
        gw = params[f"gw_{horizon}"]
        gz = params[f"gz_{horizon}"]

        rhs = gw @ weights_h + gz * zeta
        aux = [theta >= rhs]

        return -theta, aux

    def update(
        self,
        spec: CVaRCuttingPlane,
        params: dict[str, Any],
        inputs: dict[str, Any],
    ) -> None:
        moments = inputs["moments"]
        probs = np.asarray(moments.scenario_probs, dtype=float)

        for h in range(moments.n_horizons):
            n = moments.n_assets
            R_h = np.asarray(moments.scenario_returns[:, h, :], dtype=float)

            if R_h.shape != (params["n_scenarios"], n):
                raise ValueError(
                    f"scenario_returns[:, {h}, :] must have shape "
                    f"({params['n_scenarios']}, {n}); got {R_h.shape}"
                )

            if not np.all(np.isfinite(R_h)):
                raise ValueError(
                    f"scenario_returns for horizon {h} contains NaN or inf."
                )

            # IMPORTANT:
            gw = np.zeros((spec.max_cuts, n))
            gz = np.ones(spec.max_cuts)

            # Anchor cut 0: empty-tail cut
            # theta >= zeta
            gw[0, :] = 0.0
            gz[0] = 1.0

            # Anchor cut 1: all-tail cut
            # theta >= -(1/alpha) * (p @ R) @ w + (1 - 1/alpha) * zeta
            gw[1, :] = -(1.0 / spec.alpha) * (probs @ R_h)
            gz[1] = 1.0 - (1.0 / spec.alpha) * probs.sum()

            params[f"gw_{h}"].value = gw
            params[f"gz_{h}"].value = gz
            params["cut_count"][h] = 2

    def _compute_actual_cvar(
        self,
        losses: NDArray[np.floating],
        zeta_val: float,
        probs: NDArray[np.floating],
        alpha: float,
    ) -> float:
        shortfall = np.maximum(losses - zeta_val, 0.0)
        return zeta_val + (1.0 / alpha) * (probs @ shortfall)

    def _build_cut(
        self,
        losses: NDArray[np.floating],
        R_h: NDArray[np.floating],
        zeta_val: float,
        probs: NDArray[np.floating],
        alpha: float,
    ) -> tuple[NDArray[np.floating], float]:
        tail_mask = losses > zeta_val
        if not np.any(tail_mask):
            tail_mask[np.argmax(losses)] = True

        tail_probs = probs * tail_mask
        g_w = -(1.0 / alpha) * (tail_probs @ R_h)
        g_zeta = 1.0 - (1.0 / alpha) * tail_probs.sum()
        return g_w, g_zeta

    def _activate_cut(
        self,
        params: dict[str, Any],
        h: int,
        g_w: NDArray[np.floating],
        g_zeta: float,
        max_cuts: int,
    ) -> None:
        if max_cuts <= 2:
            raise ValueError("max_cuts must be greater than 2.")

        # Rotate only through dynamic cuts 2, ..., max_cuts - 1.
        k = 2 + ((params["cut_count"][h] - 2) % (max_cuts - 2))

        gw_val = params[f"gw_{h}"].value
        gz_val = params[f"gz_{h}"].value

        if gw_val is None or gz_val is None:
            raise RuntimeError(
                f"CVaR cut parameters for horizon {h} are not initialized."
            )

        # Mutate the existing CVXPY parameter buffers in place.
        # Re-assigning ``.value`` here would run CVXPY's validation/copy path
        # again, even though these arrays are already the parameter backing data.
        gw_val[k, :] = np.asarray(g_w, dtype=float)
        gz_val[k] = g_zeta

        params["cut_count"][h] += 1

    def _refine_horizon(
        self,
        spec: CVaRCuttingPlane,
        params: dict[str, Any],
        w_h: NDArray[np.floating],
        R_h: NDArray[np.floating],
        probs: NDArray[np.floating],
        h: int,
    ) -> bool:
        zeta_raw = params[f"zeta_{h}"].value
        theta_raw = params[f"theta_{h}"].value

        if zeta_raw is None or theta_raw is None:
            raise RuntimeError(
                f"CVaR variables for horizon {h} have no value after solve."
            )

        zeta_val = float(np.asarray(zeta_raw).reshape(()))
        theta_val = float(np.asarray(theta_raw).reshape(()))

        losses = -R_h @ w_h
        actual_cvar = self._compute_actual_cvar(losses, zeta_val, probs, spec.alpha)

        # Only add a cut if true CVaR is materially above theta.
        gap = actual_cvar - theta_val

        if gap <= spec.tol * max(1.0, abs(actual_cvar)):
            return True

        g_w, g_zeta = self._build_cut(losses, R_h, zeta_val, probs, spec.alpha)
        self._activate_cut(params, h, g_w, g_zeta, spec.max_cuts)
        return False

    def refine(
        self,
        spec: CVaRCuttingPlane,
        params: dict[str, Any],
        weights_val: NDArray[np.floating],
        moments: HorizonMoments,
    ) -> bool:
        probs = moments.scenario_probs

        converged = True
        for h in range(moments.n_horizons):
            h_converged = self._refine_horizon(
                spec,
                params,
                weights_val[h, :],
                moments.scenario_returns[:, h, :],
                probs,
                h,
            )
            converged = converged and h_converged

        return converged


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
        _cvar_alpha_check(spec.alpha)

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
        *,
        cash_h: cp.Expression | None = None,
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
        *,
        cash_h: cp.Expression | None = None,
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

    Three components are combined:

    1. **Linear cost** (e.g. half bid-ask spread + per-share brokerage)::

           a_i * |z_i|

       where ``a_i = spec.cost + spec.pershare_cost / price_i``.
       The per-share term is only active when ``inputs["prices"]`` is provided
       and ``spec.pershare_cost > 0``.

    2. **Market-impact cost** (Almgren-style power law)::

           b * sigma_i / V_i^(p-1) * |z_i|^p

       where ``b = spec.market_impact``, ``sigma_i`` is the per-asset
       period volatility (derived from the horizon-0 covariance diagonal),
       ``V_i`` is the per-asset average daily volume (from ``inputs``), and
       ``p = spec.exponent`` (1.5 by default → square-root impact).
       Falls back to ``b * sigma_i`` when volume data is unavailable.

    3. **Directional slippage bias** (the ``c`` term)::

           c_bias * z_i

       Signed, not absolute — models systematic execution-price deviation.
       E.g. set to the negative of the expected open-to-VWAP return if
       execution happens at VWAP.  Default 0 (disabled).

    The expression is negative so that maximising it minimises costs.
    """

    def allocate(
        self, spec: TransactionCost, horizons: int, n_assets: int, **_kwargs
    ) -> dict[str, Any]:
        return {
            # Per-asset linear cost coefficient  a_i  (>= 0)
            "tc_linear": cp.Parameter(n_assets, nonneg=True, name="tc_linear"),
            # Per-asset market-impact coefficient  b * sigma_i / V_i^(p-1)  (>= 0)
            "tc_impact": cp.Parameter(n_assets, nonneg=True, name="tc_impact"),
            # Per-asset directional bias  c_i  (signed — no nonneg constraint)
            # This is a signed Parameter: buying penalised when c > 0,
            # selling penalised when c < 0.
            "tc_bias": cp.Parameter(n_assets, name="tc_bias"),
        }

    def compile(
        self,
        spec: TransactionCost,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
        *,
        cash_h: cp.Expression | None = None,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        linear = cp.sum(cp.multiply(params["tc_linear"], cp.abs(trades_h)))

        impact = cp.sum(
            cp.multiply(
                params["tc_impact"],
                cp.power(cp.abs(trades_h), spec.exponent),
            )
        )

        bias = params["tc_bias"] @ trades_h

        # All three are costs (positive = bad), so negate for maximisation.
        return (-1.0 * (linear + impact + bias), [])  # type: ignore[return-value]

    def update(
        self, spec: TransactionCost, params: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        moments = inputs["moments"]
        n = moments.n_assets

        linear_coeff = np.full(n, spec.cost)

        if spec.pershare_cost > 0.0:
            prices: NDArray[np.floating] | None = inputs.get("prices")
            if prices is not None and np.all(prices > 0):
                linear_coeff = linear_coeff + spec.pershare_cost / np.asarray(prices)
            else:
                pass

        params["tc_linear"].value = np.maximum(linear_coeff, 0.0)

        sigma = np.sqrt(np.maximum(np.diag(moments.covariances[0]), 0.0))

        volume: NDArray[np.floating] | None = inputs.get("volume")
        if volume is not None and np.all(volume > 0):
            vol_factor = volume ** (spec.exponent - 1.0)
            impact_coeff = spec.market_impact * sigma / vol_factor
        else:
            impact_coeff = spec.market_impact * sigma

        params["tc_impact"].value = np.maximum(impact_coeff, 0.0)

        params["tc_bias"].value = np.broadcast_to(
            np.asarray(spec.c_bias, dtype=float), (n)
        ).copy()


@register_objective(HoldingCost)
class HoldingCostHandler:
    """
    Penalise overnight holding costs at each horizon.

    The per-period objective contribution is::

        - short_rate * sum(neg(w_h))   # borrow fee on shorts
        - long_rate  * sum(pos(w_h))   # custody / financing on longs
        + div_rate   * sum(w_h)        # dividend income (if using price returns)

    where ``*_rate = annual_pct / (100 * periods_per_year)``.

    DCP validity
    ------------
    ``cp.neg(w)`` and ``cp.pos(w)`` are convex in ``w``.
    Multiplying by a nonneg parameter and negating gives a concave term,
    which is valid in a maximisation problem.  ``div_rate * sum(w)`` is
    affine — always valid.

    DPP validity
    ------------
    All coefficients are ``cp.Parameter`` objects (not Python scalars),
    so the symbolic graph is fixed; only the parameter values change between
    solves.  This satisfies CVXPY's DPP rules and enables warm-starting.
    """

    def allocate(
        self, spec: HoldingCost, horizons: int, n_assets: int, **_kwargs
    ) -> dict[str, Any]:
        # Scalar parameters: one rate applies uniformly to all assets.
        # nonneg=True on fee params lets CVXPY verify the concavity argument
        # automatically (-nonneg_param * convex_expr is concave).
        return {
            "short_rate": cp.Parameter(nonneg=True, name="hc_short_rate"),
            "long_rate": cp.Parameter(nonneg=True, name="hc_long_rate"),
            "div_rate": cp.Parameter(name="hc_div_rate"),  # signed: income
        }

    def compile(
        self,
        spec: HoldingCost,
        params: dict[str, Any],
        weights_h: cp.Expression,
        trades_h: cp.Expression,
        horizon: int,
        *,
        cash_h: cp.Expression | None = None,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        short_rate = params["short_rate"]
        long_rate = params["long_rate"]
        div_rate = params["div_rate"]

        short_cost = short_rate * cp.sum(cp.neg(weights_h))
        long_cost = long_rate * cp.sum(cp.pos(weights_h))

        # Dividend income: affine in w, always DCP + DPP.
        div_income = div_rate * cp.sum(weights_h)

        return (-short_cost - long_cost + div_income, [])

    def update(
        self, spec: HoldingCost, params: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """
        Converts annualised percentages to per-period rates and fills
        the CVXPY parameters.

        No ``inputs`` keys are required — all rates come from the spec.
        """
        ppy = spec.periods_per_year

        params["short_rate"].value = spec.short_fees / (100.0 * ppy)
        params["long_rate"].value = spec.long_fees / (100.0 * ppy)
        params["div_rate"].value = spec.dividends / (100.0 * ppy)
