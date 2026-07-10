from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import polars as pl
from polars import DataFrame

from qraft.core.configs import (
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    PipelineConfig,
    SamplingMethod,
    SimulationForecastConfig,
)
from qraft.core.panel import ScenarioPanel
from qraft.core.probability.sampling import weighted_bootstrapping_idx
from qraft.core.scenarios.copula_marginal import CMAConfig, CopulaMarginalModel
from qraft.core.universe import AssetUniverse
from qraft.forecast.forecast_paths import ForecastPaths, InnovationPaths
from qraft.forecast.pipelines.fitted_universe import (
    FittedUniverse,
    apply_forecast_recipe,
    create_forecast_recipe,
    fit_diagnostics,
)
from qraft.forecast.time_series.transforms.inverses import apply_inverse_transforms
from qraft.utils.log import warning_event

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

    logger.debug(
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

        logger.debug("CMA update complete: n_scenarios=%d", len(invariants))

    invariants_vector = invariants.values.to_numpy()
    prob = invariants.prob

    if method == "historical":
        logger.debug("Returning historical innovations without resampling")
        return InnovationPaths(
            values=invariants_vector[:, None, :],
            path_probs=prob,
        )

    n_draws = n_sims * horizon
    logger.debug("Bootstrapping %d innovation draws", n_draws)

    idx = weighted_bootstrapping_idx(
        invariants.values,
        prob,
        n_samples=n_draws,
        seed=seed,
    )
    simulated_draws = invariants_vector[idx].reshape(n_sims, horizon, len(assets))

    logger.debug("Innovation draw complete with output shape=%s", simulated_draws.shape)
    return InnovationPaths(
        values=simulated_draws,
        path_probs=np.full(n_sims, 1.0 / n_sims),  # uniform for MC
    )


def _forecast_dates_from_history(dates: pl.Series, horizon: int) -> pl.Series:
    if len(dates) < 2:
        raise ValueError("run_forecast requires at least two dates to infer cadence")

    previous_date = dates[-2]
    last_date = dates[-1]
    step = last_date - previous_date
    if step <= timedelta(0):
        raise ValueError("ScenarioPanel dates must be strictly increasing")

    return pl.Series("date", [last_date + step * i for i in range(1, horizon + 1)])


def _model_name(order: object, distribution: object | None = None) -> str | None:
    if order is None:
        return None
    if distribution is None:
        return str(order)
    return f"{order} {distribution}"


def _model_health_frame(fit: FittedUniverse) -> DataFrame:
    recipe = fit.recipe()
    diagnostics = fit_diagnostics(fit)
    rows = []
    for asset in fit.assets:
        quality = recipe.quality.get(asset)
        values = diagnostics.get(asset, {})
        rows.append(
            {
                "asset": asset,
                "mean_model": _model_name(recipe.mean_orders.get(asset)),
                "vol_model": _model_name(
                    recipe.vol_orders.get(asset), recipe.vol_distributions.get(asset)
                ),
                "quality_grade": getattr(quality, "grade", None),
                "quality_score": getattr(quality, "score", None),
                "fallback_reason": values.get("fallback_reason"),
                "cap_bind_rate": values.get("bind_rate"),
                "admissible": values.get("admissible"),
            }
        )
    return DataFrame(rows)


def forecast_from_fit(
    panel: ScenarioPanel,
    asset_universe: AssetUniverse,
    universe_fit: FittedUniverse,
    last_data: DataFrame,
    n_rows: int,
    seed: int | None = None,
    include_fit_diagnostics: bool = False,
    *,
    simulation_config: SimulationForecastConfig,
) -> ForecastPaths:
    innovations = draw_innovations(
        invariants=universe_fit.invariants,
        horizon=simulation_config.horizon,
        n_sims=simulation_config.n_sims,
        seed=seed,
        method=simulation_config.method,
        cma_config=simulation_config.cma_config,
    )

    simulated = universe_fit.simulate(innovations.values)

    logger.debug("Forecast complete - applying inverse transforms")
    transformed = apply_inverse_transforms(
        asset_data_dict=simulated,
        n_original=n_rows,
        inverse_specs=universe_fit.inverse_specs,
        back_to_price=simulation_config.back_to_price,
    )

    forecast_assets = list(transformed.keys())
    n_dropped = len(asset_universe.all_tickers) - len(forecast_assets)
    if n_dropped > 0:
        warning_event(
            logger,
            "forecast.asset_dropped",
            "Forecast dropped assets from original universe",
            forecast_assets=len(forecast_assets),
            dropped=n_dropped,
            original_assets=len(asset_universe.all_tickers),
            remaining=tuple(forecast_assets),
        )

    initial_prices = {
        col: np.exp((last_data[col][0]))
        for col in forecast_assets
        if col in last_data.columns
    }

    filtered_factors = [f for f in asset_universe.factors if f in forecast_assets]
    filtered_assets = [a for a in asset_universe.assets if a in forecast_assets]
    forecast_universe = AssetUniverse(assets=filtered_assets, factors=filtered_factors)

    return ForecastPaths(
        asset_paths=transformed,
        dates=_forecast_dates_from_history(panel.dates, simulation_config.horizon),
        path_probs=innovations.path_probs,
        universe=forecast_universe,
        initial_prices=initial_prices,
        diagnostics=None if not include_fit_diagnostics else universe_fit,
        model_health_frame=_model_health_frame(universe_fit),
    )


def run_forecast(
    panel: ScenarioPanel,
    universe: AssetUniverse,
    seed: int | None = None,
    include_fit_diagnostics: bool = False,
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

    recipe = create_forecast_recipe(
        data=data,
        prob=prob,
        universe=universe,
        seed=seed,
        pipeline_config=pipeline_config,
    )
    universe_fit = apply_forecast_recipe(recipe, data, prob, pipeline_config)
    return forecast_from_fit(
        panel=panel,
        asset_universe=universe,
        universe_fit=universe_fit,
        last_data=data.tail(1),
        n_rows=data.height,
        seed=seed,
        include_fit_diagnostics=include_fit_diagnostics,
        simulation_config=simulation_config,
    )
