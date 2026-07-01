from __future__ import annotations

import numpy as np
from qraft.forecast.simulation.engines.mean import mean_simulation_paths
from qraft.forecast.simulation.engines.utils import as_sims_by_horizon
from qraft.forecast.simulation.engines.volatility import garch_simulation_paths
from qraft.forecast.simulation.state import SimulationForecast
from numpy._typing import NDArray


def simulate_asset_paths(
    forecast_model: SimulationForecast,
    innovations: NDArray[np.floating],
    asset: str | None = None,
) -> NDArray[np.floating]:
    """
    Simulate asset paths given a prepared simulation forecast and innovations.

    This function adapts the provided ``innovations`` into a consistent
    (n_sims, horizon, ) layout and dispatches to the appropriate volatility
    and mean simulation engines based on the forecast model configuration.

    Parameters
    ----------
    forecast_model : SimulationForecast
        Prepared simulation forecast containing a :class:`UnivariateModel`
        and the initial state (lags/histories) required by the simulation.
    innovations : NDArray[np.floating]
        Innovations array. Accepted shapes are one of:
          - (n_scenarios, )
          - (n_sims, horizon)
          - (n_sims, horizon,)
        The utility ``as_sims_by_horizon`` is used to normalize shapes into
        ``(n_sims, horizon, )``.

    Returns
    -------
    NDArray[np.floating]
        Simulated paths array with shape ``(n_sims, horizon)`` containing the
        simulated series (in transformed/invariant space) for the asset.

    Raises
    ------
    ValueError
        If the model uses an unknown ``vol_kind``.
    """
    model = forecast_model.model
    params = model.compile_params()
    state0 = forecast_model.state0

    innovations, n_sims, horizon = as_sims_by_horizon(innovations)

    if model.vol_kind == "none":
        eps_paths = model.innovation_scale * innovations
    elif model.vol_kind == "garch":
        _, eps_paths, diagnostics = garch_simulation_paths(
            params=params,
            garch_order=model.vol_order,
            eps_start=state0.vol_residual_lags,
            var_start=state0.var_hist,
            innovations=innovations,
            var_cap=state0.var_cap,
            unconditional_variance=state0.unconditional_variance,
            asset=asset,
        )
        forecast_model.variance_cap_diagnostics = diagnostics.as_dict()
    else:
        raise ValueError(f"Unknown vol_kind: {model.vol_kind}")

    y_paths = mean_simulation_paths(
        params=params,
        mean_kind=model.mean_kind,
        mean_order=model.mean_order,
        state_series_hist=state0.series_hist,
        state_ma_resid_lags=state0.ma_residual_lags,
        eps_paths=eps_paths,
    )

    return y_paths
