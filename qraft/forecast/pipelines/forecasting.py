from __future__ import annotations

import logging

import numpy as np

from qraft.core.configs import (
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    IIDConfig,
    PipelineConfig,
    SamplingMethod,
    SimulationForecastConfig,
)
from qraft.core.panel import ScenarioPanel
from qraft.core.probability.sampling import weighted_bootstrapping_idx
from qraft.core.scenarios.copula_marginal import CMAConfig, CopulaMarginalModel
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths, InnovationPaths
from qraft.forecast.pipelines.fitted_universe import FittedUniverse
from qraft.forecast.time_series.preprocessing.white_noise import test_non_idd
from qraft.forecast.time_series.transforms.inverses import apply_inverse_transforms

logger = logging.getLogger(__name__)


def _validate_method_options(
    method: SamplingMethod, horizon: int, universe: AssetUniverse
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


def _test_invariants_iid(
    invariants: ScenarioPanel, iid_config: IIDConfig, seed: int | None
) -> None:
    assets = invariants.asset_names
    non_iid_invariants = test_non_idd(
        data=invariants.values,
        prob=invariants.prob,
        assets=assets,
        cfg=iid_config,
        seed=seed,
    )

    if non_iid_invariants:
        logger.warning(
            "Invariance Check: %d assets out of %d did not pass the invariant iid tests. "
            "Consider a richer mean/vol model for the list below:\n%s",
            len(non_iid_invariants),
            len(assets),
            non_iid_invariants,
        )


def draw_innovations(
    invariants: ScenarioPanel,
    horizon: int,
    n_sims: int,
    seed: int | None,
    method: SamplingMethod = "bootstrap",
    *,
    cma_config: CMAConfig | None = None,
) -> InnovationPaths:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if invariants.kind != "invariant":
        raise ValueError(
            f"draw_innovations requires a ScenarioPanel with kind='invariant'; "
            f"got kind={invariants.kind!r}"
        )

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
            seed=seed, cfg=cma_config, use_weighted_fit=True
        )

        logger.info("CMA update complete: n_scenarios=%d", len(invariants))

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


def _check_missing_assets(new_assets: list[str], old_assets: list[str]) -> bool:
    return bool(set(new_assets) - set(old_assets))


def run_forecast(
    panel: ScenarioPanel,
    universe: AssetUniverse,
    seed: int | None = None,
    *,
    simulation_config: SimulationForecastConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> ForecastPaths:
    if pipeline_config is None:
        pipeline_config = DEFAULT_PIPELINE_CONFIG
    if simulation_config is None:
        simulation_config = DEFAULT_SIMULATION_CONFIG

    if panel.kind != "log_price":
        raise ValueError(
            f"run_forecast requires a ScenarioPanel with kind='log_price'; "
            f"got kind={panel.kind!r}. Use ScenarioPanel.from_prices() or "
            "ScenarioPanel.from_log_prices() for forecast inputs."
        )

    _validate_method_options(
        simulation_config.method, simulation_config.horizon, universe
    )

    data = panel.to_frame()
    prob = panel.prob

    universe_fit = FittedUniverse.fit(
        data=data,
        prob=prob,
        assets=universe.all_tickers,
        seed=seed,
        pipeline_config=pipeline_config,
    )

    _test_invariants_iid(
        invariants=universe_fit.invariants,
        iid_config=DEFAULT_PIPELINE_CONFIG.preprocess.iid,
        seed=seed,
    )

    innovations = draw_innovations(
        invariants=universe_fit.invariants,
        horizon=simulation_config.horizon,
        n_sims=simulation_config.n_sims,
        seed=seed,
        method=simulation_config.method,
        cma_config=simulation_config.cma_config,
    )

    simulated = universe_fit.simulate(innovations.values)

    logger.info("Forecast complete - applying inverse transforms")
    transformed = apply_inverse_transforms(
        asset_data_dict=simulated,
        n_original=data.height,
        inverse_specs=universe_fit.inverse_specs,
        back_to_price=simulation_config.back_to_price,
    )

    if _check_missing_assets(universe.all_tickers, list(transformed.keys())):
        raise RuntimeError("Forecast output is missing assets")

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
