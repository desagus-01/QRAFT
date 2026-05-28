from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pipelines.model_selection import get_univariate_results
from pipelines.preprocess import run_univariate_preprocess
from policy import PipelineConfig
from scenarios.panel import ScenarioPanel
from scenarios.types import ProbVector
from simulation.simulate_paths import simulate_asset_paths
from simulation.state import SimulationForecast
from time_series.models.fitted_types import UnivariateRes
from time_series.preprocessing.types import UnivariatePreprocess
from time_series.transforms.inverses import InverseSpec

logger = logging.getLogger(__name__)


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
        assets: list[str],
        cfg: PipelineConfig | None = None,
    ) -> FittedUniverse:
        """Run preprocess + model selection and assemble the invariants panel."""
        preprocess = run_univariate_preprocess(
            data=data,
            prob=prob,
            assets=assets,
            cfg=cfg.preprocess if cfg is not None else None,
        )
        logger.info(
            "Preprocessing complete: post_data_shape=%s assets_to_model=%s",
            preprocess.post_data.shape,
            preprocess.needs_further_modelling,
        )

        models = get_univariate_results(
            data=preprocess.post_data,
            assets_to_model=preprocess.needs_further_modelling,
            cfg=cfg,
        )
        logger.info(
            "Univariate model selection complete for %d assets",
            len(preprocess.needs_further_modelling),
        )

        simulation_forecasts = _build_simulation_forecasts(
            data=preprocess.post_data,
            models=models,
            assets=assets,
        )

        invariants = _build_invariants_panel(
            post=preprocess.post_data,
            models=models,
            assets=assets,
            prob=prob,
        )

        return cls(
            assets=list(assets),
            preprocess=preprocess,
            models=models,
            simulation_forecasts=simulation_forecasts,
            invariants=invariants,
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


def _build_simulation_forecasts(
    data: pl.DataFrame,
    models: Mapping[str, UnivariateRes],
    assets: list[str],
) -> dict[str, SimulationForecast]:
    """Build a per-asset ``SimulationForecast`` (model + seeded state)."""
    forecasts: dict[str, SimulationForecast] = {}
    for asset in assets:
        series = data.select(asset).drop_nulls().to_numpy().ravel()
        forecasts[asset] = SimulationForecast.from_res_and_series(
            fitting_results=models[asset],
            post_series_non_null=series,
        )
    return forecasts


def _build_invariants_panel(
    post: pl.DataFrame,
    models: Mapping[str, UnivariateRes],
    assets: list[str],
    prob: ProbVector,
) -> ScenarioPanel:
    """Assemble an :class:`AssetPanel` of invariants aligned to ``post`` dates.

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
        invariant = model.invariant(values)
        patches.append(used_dates.with_columns(pl.Series(asset, invariant)))

    innovations_df = base
    for patch in patches:
        innovations_df = innovations_df.join(patch, on="date", how="left")

    logger.info("Invariants shape=%s", innovations_df.shape)
    return ScenarioPanel.from_frame(innovations_df, prob)
