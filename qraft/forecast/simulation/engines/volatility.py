from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

from qraft.forecast.simulation.engines.utils import (
    as_sims_by_horizon,
    broadcast_last_k_lags,
    garch_params,
    lag_matrix,
)
from qraft.forecast.time_series.models.model_types import CompiledParams

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VarianceCapDiagnostics:
    cap_enabled: bool
    cap_value: float | None
    unconditional_variance: float | None = None
    bind_count: int = 0
    step_count: int = 0
    max_raw_variance: float | None = None

    @property
    def bind_rate(self) -> float:
        return self.bind_count / self.step_count if self.step_count else 0.0

    @property
    def bound(self) -> bool:
        return self.bind_count > 0

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "cap_enabled": self.cap_enabled,
            "cap_value": self.cap_value,
            "unconditional_variance": self.unconditional_variance,
            "bind_count": self.bind_count,
            "step_count": self.step_count,
            "bind_rate": self.bind_rate,
            "max_raw_variance": self.max_raw_variance,
            "bound": self.bound,
        }


@dataclass
class GarchSimulator:
    omega: float
    alpha: NDArray[np.floating]
    gamma: NDArray[np.floating]
    beta: NDArray[np.floating]
    p: int
    o: int
    q: int
    eps_ext: NDArray[np.floating]
    var_ext: NDArray[np.floating]
    eps_lag: int
    var_lag: int
    var_floor: float = 1e-12
    var_cap: float | None = None
    diagnostics: VarianceCapDiagnostics | None = None

    @classmethod
    def from_state(
        cls,
        *,
        params: CompiledParams,
        order: tuple[int, int, int],
        n_sims: int,
        horizon: int,
        eps_start: NDArray[np.floating] | None,
        var_start: NDArray[np.floating] | None,
        var_cap: float | None = None,
        unconditional_variance: float | None = None,
    ) -> GarchSimulator:
        """
        Initialize a GarchSimulator from compiled parameters and starting lags.

        Parameters
        ----------
        params : CompiledParams
            Compiled GARCH parameters extracted from fitted model.
        order : tuple[int, int, int]
            (p, o, q) GARCH orders: ARCH lags p, leverage o, GARCH lags q.
        n_sims : int
            Number of parallel simulation paths.
        horizon : int
            Forecast horizon length.
        eps_start : NDArray[np.floating] | None
            Initial residual lags to seed ARCH terms.
        var_start : NDArray[np.floating] | None
            Initial variance lags to seed GARCH terms.

        Returns
        -------
        GarchSimulator
            Initialized simulator with buffers prefilled for recursion.
        """
        p, o, q = order
        omega, alpha, gamma, beta = garch_params(params, order)

        eps_lag = max(p, o)
        var_lag = q

        eps_ext = np.empty((n_sims, eps_lag + horizon), dtype=float)
        var_ext = np.empty((n_sims, var_lag + horizon), dtype=float)

        if eps_lag > 0:
            eps_ext[:, :eps_lag] = broadcast_last_k_lags(
                eps_start, n_sims, eps_lag, name="eps"
            )
        if var_lag > 0:
            var_ext[:, :var_lag] = broadcast_last_k_lags(
                var_start, n_sims, var_lag, name="variance"
            )

        return cls(
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            p=p,
            o=o,
            q=q,
            eps_ext=eps_ext,
            var_ext=var_ext,
            eps_lag=eps_lag,
            var_lag=var_lag,
            var_cap=var_cap,
            diagnostics=VarianceCapDiagnostics(
                cap_enabled=var_cap is not None,
                cap_value=var_cap,
                unconditional_variance=unconditional_variance,
            ),
        )

    def variance_step(self, t: int) -> NDArray[np.floating]:
        """
        Compute the next-step conditional variance for each simulation.

        Parameters
        ----------
        t : int
            Current timestep index (0-based within the forecast horizon).

        Returns
        -------
        NDArray[np.floating]
            Vector of variances (n_sims,) for the next time step.
        """
        n_sims = self.eps_ext.shape[0]
        v_next = np.full(n_sims, self.omega, dtype=float)

        eps_last = self.eps_lag + t - 1
        var_last = self.var_lag + t - 1

        if self.p > 0:
            eps_lags = lag_matrix(self.eps_ext, eps_last, self.p)
            v_next += (eps_lags * eps_lags) @ self.alpha

        if self.o > 0:
            eps_lags_o = lag_matrix(self.eps_ext, eps_last, self.o)
            ind = (eps_lags_o < 0.0).astype(float)
            v_next += (ind * (eps_lags_o * eps_lags_o)) @ self.gamma

        if self.q > 0:
            var_lags = lag_matrix(self.var_ext, var_last, self.q)
            v_next += var_lags @ self.beta

        if self.diagnostics is not None:
            self.diagnostics.step_count += n_sims
            raw_max = float(np.max(v_next))
            if self.diagnostics.max_raw_variance is None:
                self.diagnostics.max_raw_variance = raw_max
            else:
                self.diagnostics.max_raw_variance = max(
                    self.diagnostics.max_raw_variance, raw_max
                )
            if self.var_cap is not None:
                self.diagnostics.bind_count += int(
                    np.count_nonzero(v_next > self.var_cap)
                )
        return np.clip(v_next, self.var_floor, self.var_cap)

    def push(
        self, t: int, eps_next: NDArray[np.floating], var_next: NDArray[np.floating]
    ) -> None:
        """
        Store computed residuals and variances into the internal buffers.

        Parameters
        ----------
        t : int
            Current timestep index (0-based within the forecast horizon).
        eps_next : NDArray[np.floating]
            Residuals computed for the current time step (n_sims,).
        var_next : NDArray[np.floating]
            Variances computed for the current time step (n_sims,).
        """
        if self.eps_lag > 0:
            self.eps_ext[:, self.eps_lag + t] = eps_next
        if self.var_lag > 0:
            self.var_ext[:, self.var_lag + t] = var_next


def garch_simulation_paths(
    *,
    params: CompiledParams,
    garch_order: tuple[int, int, int],
    eps_start: NDArray[np.floating] | None,
    var_start: NDArray[np.floating] | None,
    innovations: NDArray[np.floating],
    var_cap: float | None = None,
    unconditional_variance: float | None = None,
    asset: str | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], VarianceCapDiagnostics]:
    """
    Simulate GARCH volatility and residual (eps) paths given innovations.

    This routine returns both conditional variances (sigma2) and scaled
    residuals (eps) paths for the provided innovations. The innovations
    are treated as standardized shocks which are scaled by the simulated
    standard deviation at each step.

    Parameters
    ----------
    params : CompiledParams
        Compiled parameters for the GARCH model.
    garch_order : tuple[int, int, int]
        (p, o, q) orders for the GARCH specification.
    eps_start : NDArray[np.floating] | None
        Initial residual lags for seeding ARCH terms.
    var_start : NDArray[np.floating] | None
        Initial variance lags for seeding GARCH terms.
    innovations : NDArray[np.floating]
        Innovations array which will be normalized to (n_sims, horizon).

    Returns
    -------
    tuple
        (sigma2, eps) where sigma2 is the simulated conditional variance array
        (n_sims, horizon) and eps is the scaled residuals array (n_sims, horizon).
    """
    innovations, n_sims, horizon = as_sims_by_horizon(innovations)

    sim = GarchSimulator.from_state(
        params=params,
        order=garch_order,
        n_sims=n_sims,
        horizon=horizon,
        eps_start=eps_start,
        var_start=var_start,
        var_cap=var_cap,
        unconditional_variance=unconditional_variance,
    )

    sigma2 = np.empty((n_sims, horizon), dtype=float)
    eps = np.empty((n_sims, horizon), dtype=float)

    for t in range(horizon):
        v_next = sim.variance_step(t)
        eps_next = np.sqrt(v_next) * innovations[:, t]

        sigma2[:, t] = v_next
        eps[:, t] = eps_next

        sim.push(t, eps_next, v_next)

    diagnostics = sim.diagnostics or VarianceCapDiagnostics(
        cap_enabled=var_cap is not None,
        cap_value=var_cap,
        unconditional_variance=unconditional_variance,
    )
    if diagnostics.bound:
        logger.warning(
            "Variance cap bound during GARCH simulation: asset=%s cap=%s "
            "bind_count=%d step_count=%d bind_rate=%.6f max_raw_variance=%s",
            asset,
            diagnostics.cap_value,
            diagnostics.bind_count,
            diagnostics.step_count,
            diagnostics.bind_rate,
            diagnostics.max_raw_variance,
        )

    return sigma2, eps, diagnostics
