from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qraft.core import metrics
from qraft.forecast.pipelines.fitted_universe import FittedUniverse


@dataclass(frozen=True, slots=True)
class VarianceCapComparison:
    asset: str
    bind_rate: float
    capped_annualized_vol: float
    uncapped_annualized_vol: float
    annualized_vol_delta: float
    capped_cvar_5: float
    uncapped_cvar_5: float
    cvar_5_delta: float


def compare_capped_vs_uncapped_simulation_risk(
    fit: FittedUniverse,
    innovation_paths: NDArray[np.floating],
    *,
    periods_per_year: float,
) -> list[VarianceCapComparison]:
    """Measure simulated risk impact from configured variance caps.

    The provided fit is simulated twice on the same innovations: once with its
    current caps and once with all upper caps disabled. The variance floor still
    applies in both runs.
    """
    original_caps = {
        asset: forecast.state0.var_cap
        for asset, forecast in fit.simulation_forecasts.items()
    }

    capped = fit.simulate(innovation_paths)
    diagnostics = {
        asset: dict(forecast.variance_cap_diagnostics)
        for asset, forecast in fit.simulation_forecasts.items()
    }

    try:
        for forecast in fit.simulation_forecasts.values():
            forecast.state0.var_cap = None
            forecast.variance_cap_diagnostics = {}
        uncapped = fit.simulate(innovation_paths)
    finally:
        for asset, cap in original_caps.items():
            fit.simulation_forecasts[asset].state0.var_cap = cap

    out: list[VarianceCapComparison] = []
    for asset in fit.assets:
        capped_returns = np.asarray(capped[asset], dtype=float).ravel()
        uncapped_returns = np.asarray(uncapped[asset], dtype=float).ravel()
        capped_vol = metrics.annualised_vol(capped_returns, periods_per_year)
        uncapped_vol = metrics.annualised_vol(uncapped_returns, periods_per_year)
        capped_cvar = float(
            metrics.cvar(capped_returns, prob=None, alpha=0.05, distribution_type="pnl")
        )
        uncapped_cvar = float(
            metrics.cvar(
                uncapped_returns, prob=None, alpha=0.05, distribution_type="pnl"
            )
        )
        out.append(
            VarianceCapComparison(
                asset=asset,
                bind_rate=float(diagnostics.get(asset, {}).get("bind_rate") or 0.0),
                capped_annualized_vol=capped_vol,
                uncapped_annualized_vol=uncapped_vol,
                annualized_vol_delta=uncapped_vol - capped_vol,
                capped_cvar_5=capped_cvar,
                uncapped_cvar_5=uncapped_cvar,
                cvar_5_delta=uncapped_cvar - capped_cvar,
            )
        )
    return out
