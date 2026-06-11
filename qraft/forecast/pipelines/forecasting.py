from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.sampling import weighted_bootstrapping_idx
from qraft.core.scenarios.copula_marginal import CMAConfig, CopulaMarginalModel
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths, InnovationPaths
from qraft.forecast.pipelines.fitted_universe import FittedUniverse
from qraft.forecast.time_series.transforms.inverses import apply_inverse_transforms

logger = logging.getLogger(__name__)

Method = Literal["bootstrap", "historical", "cma"]


def _validate_method_options(
    method: Method, horizon: int, universe: AssetUniverse
) -> None:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if horizon > 1 and method == "historical":
        raise ValueError(
            "Historical method for innovations can only be used for one-step forecasts."
        )
    if len(universe.all_tickers) <= 1 and method == "cma":
        raise ValueError(
            "Must have more than one asset in order to use the copula method."
        )


def draw_innovations(
    invariants: ScenarioPanel,
    horizon: int,
    n_sims: int,
    seed: int | None,
    method: Method = "bootstrap",
    *,
    cma_config: CMAConfig | None = None,
) -> InnovationPaths:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    assets = invariants.asset_names

    logger.info(
        "Drawing innovations with method=%s, horizon=%d, n_sims=%d, assets=%s",
        method,
        horizon,
        n_sims,
        assets,
    )

    if method == "cma":
        if cma_config is None:
            raise ValueError("method=cma requires a CMAConfig")

        invariants = CopulaMarginalModel.from_panel(invariants).update_distribution(
            seed=seed,
            cfg=cma_config,
        )

        logger.info("CMA update complete: n_scenarios=%d", invariants.n_rows)

    invariants_vector = invariants.values.to_numpy()
    prob = invariants.prob

    if method == "historical":
        logger.info("Returning historical innovations without resampling")
        return InnovationPaths(
            values=invariants_vector[:, None, :],
            path_probs=prob,
        )

    n_draws = n_sims * horizon
    logger.info("Bootstrapping %d innovation draws", n_draws)

    idx = weighted_bootstrapping_idx(
        invariants.values,
        prob,
        n_samples=n_draws,
        seed=seed,
    )
    simulated_draws = invariants_vector[idx].reshape(n_sims, horizon, len(assets))

    logger.info("Innovation draw complete with output shape=%s", simulated_draws.shape)
    return InnovationPaths(
        values=simulated_draws,
        path_probs=np.full(n_sims, 1.0 / n_sims),  # uniform for MC
    )


def run_forecast(
    panel: ScenarioPanel,
    universe: AssetUniverse,
    horizon: int = 10,
    n_sims: int = 1000,
    seed: int | None = None,
    method: Method = "bootstrap",
    *,
    back_to_price: bool = True,
    cma_config: CMAConfig | None = None,
) -> ForecastPaths:
    _validate_method_options(method, horizon, universe)
    data = panel.to_frame()
    prob = panel.prob

    universe_fit = FittedUniverse.fit(data=data, prob=prob, assets=universe.all_tickers)

    innovations = draw_innovations(
        invariants=universe_fit.invariants,
        horizon=horizon,
        n_sims=n_sims,
        seed=seed,
        method=method,
        cma_config=cma_config,
    )

    simulated = universe_fit.simulate(innovations.values)

    logger.info("Forecast complete - applying inverse transforms")
    transformed = apply_inverse_transforms(
        asset_data_dict=simulated,
        n_original=data.height,
        inverse_specs=universe_fit.inverse_specs,
        back_to_price=back_to_price,
    )

    last_row = data.tail(1)
    initial_prices = {
        col: np.exp((last_row[col][0]))
        for col in universe.all_tickers
        if col in last_row.columns
    }

    return ForecastPaths(
        asset_paths=transformed,
        path_probs=innovations.path_probs,
        universe=universe,
        initial_prices=initial_prices,
    )
