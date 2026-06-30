from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qraft.core.configs import DEFAULT_PIPELINE_CONFIG, IIDConfig, PipelineConfig
from qraft.core.panel import ScenarioPanel
from qraft.core.probability.prob_vector import ProbVector
from qraft.forecast.forecast_paths import AssetUniverse
from qraft.forecast.pipelines.model_selection import (
    _demean_fallback,
    run_univariate_pipeline,
)
from qraft.forecast.pipelines.preprocess import run_univariate_preprocess
from qraft.forecast.simulation.simulate_paths import simulate_asset_paths
from qraft.forecast.simulation.state import SimulationForecast
from qraft.forecast.time_series.models.fitted_types import (
    GARCH_DISTRIBUTIONS,
    RandomWalkRes,
    UnivariateRes,
)
from qraft.forecast.time_series.models.mean import fit_arma
from qraft.forecast.time_series.models.model_quality import ModelQuality
from qraft.forecast.time_series.models.volatility import fit_garch
from qraft.forecast.time_series.preprocessing.apply import (
    apply_deseason,
    apply_detrend,
    overwrite_with_transforms,
)
from qraft.forecast.time_series.preprocessing.types import (
    TransformDecision,
    UnivariatePreprocess,
)
from qraft.forecast.time_series.preprocessing.white_noise import test_non_idd
from qraft.forecast.time_series.transforms.inverses import (
    DifferenceInverseSpec,
    InverseSpec,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ForecastRecipe:
    detrend: dict[str, TransformDecision]
    deseason: dict[str, list[tuple[str, float]]]
    needs_modelling: list[str]
    mean_orders: dict[str, tuple[int, int] | None]
    vol_orders: dict[str, tuple[int, int, int] | None]
    mean_fallback_identity: dict[str, str | None]
    vol_distributions: dict[str, GARCH_DISTRIBUTIONS | None]
    quality: dict[str, ModelQuality | None]
    admissible: dict[str, bool | None]
    fallback_reason: dict[str, str | None]
    survivors: list[str]


@dataclass(frozen=True, slots=True)
class FittedUniverse:
    """Everything needed to simulate a joint forecast for a set of assets."""

    assets: list[str]
    preprocess: UnivariatePreprocess
    models: Mapping[str, UnivariateRes]
    simulation_forecasts: Mapping[str, SimulationForecast]
    invariants: ScenarioPanel

    @property
    def inverse_specs(self) -> dict[str, list[InverseSpec]]:
        return self.preprocess.inverse_specs

    @classmethod
    def fit(
        cls,
        data: pl.DataFrame,
        prob: ProbVector,
        universe: AssetUniverse,
        pipeline_config: PipelineConfig,
        seed: int | None = None,
    ) -> FittedUniverse:
        """Select a forecast recipe, then apply it on the same window."""
        recipe = create_forecast_recipe(
            data, prob, universe, pipeline_config, seed=seed
        )
        return apply_forecast_recipe(recipe, data, prob, pipeline_config)

    def recipe(self) -> ForecastRecipe:
        return ForecastRecipe(
            detrend=self.preprocess.detrend_decision,
            deseason=self.preprocess.deseason_decision,
            needs_modelling=list(self.preprocess.needs_further_modelling),
            mean_orders={
                a: getattr(m.mean_res, "model_order", None)
                for a, m in self.models.items()
            },
            vol_orders={
                a: getattr(m.volatility_res, "model_order", None)
                for a, m in self.models.items()
            },
            mean_fallback_identity={
                a: m.mean_res.kind if m.mean_res is not None else None
                for a, m in self.models.items()
            },
            vol_distributions={
                a: getattr(m.volatility_res, "distribution", None)
                for a, m in self.models.items()
            },
            quality={a: m.quality for a, m in self.models.items()},
            admissible={
                a: getattr(m.volatility_res, "admissible", None)
                for a, m in self.models.items()
            },
            fallback_reason={
                a: getattr(m.volatility_res, "fallback_reason", None)
                for a, m in self.models.items()
            },
            survivors=list(self.assets),
        )

    def simulate(
        self,
        innovation_paths: NDArray[np.floating],
    ) -> dict[str, NDArray[np.floating]]:
        """Run per-asset simulation given innovation paths.

        Parameters
        ----------
        innovation_paths : NDArray[np.floating]
            Shape ``(n_sims, horizon, n_assets)``. The trailing axis must
            match ``self.assets`` in order.
        """
        if innovation_paths.ndim != 3:
            raise ValueError(
                f"innovation_paths must be 3-D (n_sims, horizon, n_assets); "
                f"got ndim={innovation_paths.ndim}"
            )
        if innovation_paths.shape[2] != len(self.assets):
            raise ValueError(
                f"innovation_paths last axis ({innovation_paths.shape[2]}) "
                f"does not match number of assets ({len(self.assets)})"
            )

        logger.info("Simulating asset paths for %d assets", len(self.assets))

        paths: dict[str, NDArray[np.floating]] = {}
        for i, asset in enumerate(self.assets):
            paths[asset] = simulate_asset_paths(
                forecast_model=self.simulation_forecasts[asset],
                innovations=innovation_paths[:, :, i],
            )
        return paths


def create_forecast_recipe(
    data: pl.DataFrame,
    prob: ProbVector,
    universe: AssetUniverse,
    pipeline_config: PipelineConfig,
    seed: int | None = None,
) -> ForecastRecipe:
    """Run expensive preprocessing/model selection and return a reusable recipe."""
    all_tickers = universe.all_tickers
    preprocess = run_univariate_preprocess(
        data=data,
        prob=prob,
        assets=all_tickers,
        preprocess_config=pipeline_config.preprocess,
        seed=seed,
    )
    logger.info(
        "Preprocessing complete: post_data_shape=%s assets_to_model=%s",
        preprocess.post_data.shape,
        preprocess.needs_further_modelling,
    )

    models = run_univariate_pipeline(
        data=preprocess.post_data,
        assets_to_model=preprocess.needs_further_modelling,
        pipeline_config=pipeline_config,
    )
    logger.info(
        "Univariate model selection complete for %d assets",
        len(preprocess.needs_further_modelling),
    )

    simulation_forecasts = _build_simulation_forecasts(
        data=preprocess.post_data,
        models=models,
        assets=all_tickers,
        variance_cap_factor=pipeline_config.volatility.variance_cap_factor,
    )

    invariants = _build_invariants_panel(
        post=preprocess.post_data,
        models=models,
        assets=all_tickers,
        prob=prob,
    )
    if pipeline_config.exclude_non_invariants:
        invariants = _remove_non_invariants(
            invariants_panel=invariants,
            seed=seed,
            already_iid=preprocess.assets_already_iid,
            protected=set(universe.factors),
        )

    surviving = invariants.asset_names
    n_dropped = len(all_tickers) - len(surviving)
    if n_dropped > 0:
        models = {k: v for k, v in models.items() if k in surviving}
        simulation_forecasts = {
            k: v for k, v in simulation_forecasts.items() if k in surviving
        }
        preprocess = preprocess.filter_to(set(surviving))

        n_asset_drops = len([a for a in universe.assets if a not in surviving])
        logger.warning(
            "Removed %d non-invariant tradable asset(s) "
            "(%d factors remain protected). Surviving tickers (%d): %s",
            n_asset_drops,
            len(universe.factors),
            len(surviving),
            surviving,
        )

    fit = FittedUniverse(
        assets=list(surviving),
        preprocess=preprocess,
        models=models,
        simulation_forecasts=simulation_forecasts,
        invariants=invariants,
    )
    return fit.recipe()


def _merge_inv(
    inv_d: dict[str, InverseSpec],
    inv_s: dict[str, InverseSpec],
    survivors: list[str],
) -> dict[str, list[InverseSpec]]:
    return {
        a: [spec for spec in (inv_d.get(a), inv_s.get(a)) if spec is not None]
        for a in survivors
    }


def fit_with_orders(
    post_data: pl.DataFrame,
    recipe: ForecastRecipe,
    cfg: PipelineConfig,
) -> dict[str, UnivariateRes]:
    need = set(recipe.needs_modelling)
    out: dict[str, UnivariateRes] = {}
    for a in recipe.survivors:
        if a not in need:
            arr = post_data.select(a).drop_nulls().to_numpy().ravel()
            diff_vals = np.diff(arr)
            scale = float(np.std(diff_vals, ddof=1)) if diff_vals.size > 0 else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            mean_res = RandomWalkRes(residuals=diff_vals, residual_scale=scale)
            out[a] = UnivariateRes(
                mean_res=mean_res,
                volatility_res=None,
                quality=recipe.quality.get(a),
            )
            continue
        arr = post_data.select(a).drop_nulls().to_numpy().ravel()
        mean_order = recipe.mean_orders[a]
        if mean_order is not None:
            mean_res = fit_arma(arr, mean_order, cfg.mean)
            if mean_res is None:
                logger.warning(
                    "Asset=%s: ARMA(%d,%d) fit failed; falling back to demean model",
                    a,
                    *mean_order,
                )
                mean_res = _demean_fallback(arr)
        else:
            mean_res = _demean_fallback(arr)
        vol_order = recipe.vol_orders[a]
        if vol_order is not None and mean_res is not None:
            distribution = recipe.vol_distributions[a]
            if distribution is None:
                raise ValueError(f"Asset={a} recipe is missing GARCH distribution")
            vol_res = fit_garch(
                mean_res.residuals, vol_order, distribution, cfg.volatility
            )
        else:
            vol_res = None
        out[a] = UnivariateRes(
            mean_res=mean_res,
            volatility_res=vol_res,
            quality=recipe.quality.get(a),
        )
    return out


def apply_forecast_recipe(
    recipe: ForecastRecipe,
    data: pl.DataFrame,
    prob: ProbVector,
    cfg: PipelineConfig,
) -> FittedUniverse:
    after_detrend, inv_d = apply_detrend(data, recipe.detrend, prob)
    merged = overwrite_with_transforms(
        base=data, patch=after_detrend, assets=recipe.survivors, suffix="_detrend"
    )
    post, inv_s = apply_deseason(merged, recipe.deseason)
    missing = [a for a in recipe.survivors if a not in post.columns]
    if missing:
        post = post.join(merged.select(["date"] + missing), on="date", how="left")

    models = fit_with_orders(post, recipe, cfg)

    inverse_specs = _merge_inv(inv_d, inv_s, recipe.survivors)
    need = set(recipe.needs_modelling)
    for a in recipe.survivors:
        if a not in need:
            last_level = float(data.get_column(a).drop_nulls()[-1])
            inverse_specs[a] = [
                DifferenceInverseSpec(order=1, initial_values=np.asarray([last_level]))
            ]

    return FittedUniverse(
        assets=recipe.survivors,
        preprocess=UnivariatePreprocess(
            post_data=post,
            inverse_specs=inverse_specs,
            needs_further_modelling=[a for a in recipe.survivors if a in need],
            detrend_decision=recipe.detrend,
            deseason_decision=recipe.deseason,
        ),
        models=models,
        simulation_forecasts=_build_simulation_forecasts(
            post, models, recipe.survivors, cfg.volatility.variance_cap_factor
        ),
        invariants=_build_invariants_panel(post, models, recipe.survivors, prob),
    )


def select_recipe(
    data: pl.DataFrame,
    prob: ProbVector,
    universe: AssetUniverse,
    pipeline_config: PipelineConfig,
    seed: int | None = None,
) -> ForecastRecipe:
    return create_forecast_recipe(data, prob, universe, pipeline_config, seed=seed)


def apply_recipe(
    recipe: ForecastRecipe,
    data: pl.DataFrame,
    prob: ProbVector,
    cfg: PipelineConfig,
) -> FittedUniverse:
    return apply_forecast_recipe(recipe, data, prob, cfg)


def recondition(
    recipe: ForecastRecipe,
    data: pl.DataFrame,
    prob: ProbVector,
    cfg: PipelineConfig,
) -> FittedUniverse:
    return apply_forecast_recipe(recipe, data, prob, cfg)


def _build_simulation_forecasts(
    data: pl.DataFrame,
    models: Mapping[str, UnivariateRes],
    assets: list[str],
    variance_cap_factor: float = 4.0,
) -> dict[str, SimulationForecast]:
    forecasts: dict[str, SimulationForecast] = {}
    for asset in assets:
        series = data.select(asset).drop_nulls().to_numpy().ravel()
        forecasts[asset] = SimulationForecast.from_res_and_series(
            fitting_results=models[asset],
            post_series_non_null=series,
            variance_cap_factor=variance_cap_factor,
        )
    return forecasts


def _test_invariants_iid(
    invariants: ScenarioPanel,
    iid_config: IIDConfig,
    seed: int | None,
    already_iid_names: list[str] | None = None,
) -> list[str]:
    all_assets = invariants.asset_names
    assets_to_test = all_assets
    if already_iid_names:
        assets_to_test = [
            a for a in invariants.asset_names if a not in already_iid_names
        ]
    if not assets_to_test:
        return []
    non_iid_invariants = test_non_idd(
        data=invariants.to_frame(),
        prob=invariants.prob,
        assets=assets_to_test,
        cfg=iid_config,
        seed=seed,
    )

    if non_iid_invariants:
        logger.warning(
            "Invariance Check: %d assets out of %d did not pass the invariant iid tests. "
            "Consider a richer mean/vol model for the list below:\n%s",
            len(non_iid_invariants),
            len(all_assets),
            non_iid_invariants,
        )
    return non_iid_invariants


def _remove_non_invariants(
    invariants_panel: ScenarioPanel,
    seed: int | None,
    already_iid: list[str],
    protected: set[str] | None = None,
) -> ScenarioPanel:
    non_invariant_assets = _test_invariants_iid(
        invariants=invariants_panel,
        iid_config=DEFAULT_PIPELINE_CONFIG.preprocess.iid,
        seed=seed,
        already_iid_names=already_iid,
    )
    if protected:
        protected_failures = [a for a in non_invariant_assets if a in protected]
        if protected_failures:
            logger.warning(
                "The following %d factor(s) did not pass the invariant iid tests "
                "but are PROTECTED and will NOT be dropped: %s",
                len(protected_failures),
                protected_failures,
            )
        to_drop = [a for a in non_invariant_assets if a not in protected]
    else:
        to_drop = non_invariant_assets

    new_invariants_df = invariants_panel.values.drop(to_drop)

    return ScenarioPanel(
        values=new_invariants_df,
        dates=invariants_panel.dates,
        prob=invariants_panel.prob,
        kind="invariant",
    )


def _build_invariants_panel(
    post: pl.DataFrame,
    models: Mapping[str, UnivariateRes],
    assets: list[str],
    prob: ProbVector,
) -> ScenarioPanel:
    """
    Each column holds the invariant (innovation) series for one asset, with
    nulls preserved where the underlying post-processed series was null.
    """
    logger.info("Building invariants panel for assets=%s", assets)

    base = post.select("date")
    patches: list[pl.DataFrame] = []
    for asset in assets:
        model = models[asset]
        used_dates = post.filter(pl.col(asset).is_not_null()).select("date")
        values = post.select(asset).drop_nulls().to_numpy().ravel()

        logger.debug(
            "Asset=%s: n_values=%d, has_vol=%s, has_mean=%s",
            asset,
            values.size,
            model.volatility_res is not None,
            model.mean_res is not None,
        )

        invariant = model.invariant(values)

        patches.append(used_dates.with_columns(pl.Series(asset, invariant)))

    innovations_df = base
    for patch in patches:
        innovations_df = innovations_df.join(patch, on="date", how="left")
        new_nans = (
            innovations_df.select(pl.all().is_null().sum()).to_numpy().ravel().sum()
        )
        if new_nans > 0:
            logger.debug(
                "After left-join: %d new nulls introduced across all columns", new_nans
            )

    logger.info("Invariants shape=%s", innovations_df.shape)
    logger.debug(
        "Total nulls in innovations_df: %s",
        innovations_df.select(pl.all().is_null().sum()).row(0),
    )
    return ScenarioPanel.from_invariants(
        innovations_df,
        prob,
        drop_nulls=True,
    )
