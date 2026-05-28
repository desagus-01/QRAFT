from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

from simulation.engines.utils import (
    as_sims_by_horizon,
    broadcast_last_k_lags,
    garch_params,
    lag_matrix,
)
from time_series.models.model_types import CompiledParams


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

        return np.maximum(v_next, self.var_floor)

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
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
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
    )

    sigma2 = np.empty((n_sims, horizon), dtype=float)
    eps = np.empty((n_sims, horizon), dtype=float)

    for t in range(horizon):
        v_next = sim.variance_step(t)
        eps_next = np.sqrt(v_next) * innovations[:, t]

        sigma2[:, t] = v_next
        eps[:, t] = eps_next

        sim.push(t, eps_next, v_next)

    return sigma2, eps
