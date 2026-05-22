from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from portfolio.policy.moments import HorizonMoments
from portfolio.policy.objectives.protocol import register_objective
from portfolio.policy.objectives.specs import (
    CovarianceRisk,
    CVaRCuttingPlane,
    CVaRRisk,
    ExpectedReturn,
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
