from __future__ import annotations

import logging

from qraft.construction.market_snapshot import MarketSnapshot
from qraft.construction.optimization.moments import PolicyInputConfig, PolicyInputs
from qraft.construction.policies import MPOPolicy, PolicyDecision
from qraft.construction.state import PortfolioState
from qraft.core.configs import (
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    PipelineConfig,
    SimulationForecastConfig,
)
from qraft.forecast.forecast_paths import AssetUniverse, ForecastPaths
from qraft.forecast.pipelines.fitted_universe import (
    FittedUniverse,
    ForecastRecipe,
    recondition,
)
from qraft.forecast.pipelines.forecasting import _forecast_from_fit

logger = logging.getLogger(__name__)


class BacktestForecaster:
    """Owns the forecast cadence and cache for a backtest run.

    Three cadences, counted in rebalances (decision bars):
      * ``recipe_every`` (R): rerun full model selection -> ForecastRecipe
        (order selection, transform decisions, survivor set).
      * ``forecast_every`` (N <= R): recondition the cached recipe on the
        latest window and regenerate ForecastPaths -> PolicyInputs.
      * optimization runs every rebalance in the simulator, reusing the most
        recently cached PolicyInputs against the live portfolio state.
    """

    def __init__(
        self,
        input_config: PolicyInputConfig,
        *,
        recipe_every: int = 12,
        forecast_every: int = 1,
        pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
        simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
        seed: int | None = None,
    ) -> None:
        if recipe_every < 1 or forecast_every < 1:
            raise ValueError("recipe_every and forecast_every must be >= 1")
        if forecast_every > recipe_every:
            raise ValueError("forecast_every (N) must be <= recipe_every (R)")
        self.input_config = input_config
        self.recipe_every = recipe_every
        self.forecast_every = forecast_every
        self.pipeline_config = pipeline_config
        self.simulation_config = simulation_config
        self.seed = seed

        self._recipe: ForecastRecipe | None = None
        self._cached_inputs: PolicyInputs | None = None
        self._last_universe: tuple[str, ...] | None = None

    def policy_inputs_at(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
        """Return PolicyInputs for rebalance ``step`` (0-based)."""
        universe_key = tuple(snapshot.universe.all_tickers)
        universe_changed = universe_key != self._last_universe

        rebuild_recipe = (
            self._recipe is None or universe_changed or step % self.recipe_every == 0
        )
        reforecast = (
            rebuild_recipe
            or self._cached_inputs is None
            or step % self.forecast_every == 0
        )

        if not reforecast:
            assert self._cached_inputs is not None
            return self._cached_inputs

        panel = snapshot.history
        universe = snapshot.universe
        data = panel.to_frame()
        prob = panel.prob
        seed = self._seed_for(step)

        if rebuild_recipe:
            fit = FittedUniverse.fit(
                data=data,
                prob=prob,
                universe=universe,
                pipeline_config=self.pipeline_config,
                seed=seed,
            )
            self._recipe = fit.recipe()
            self._last_universe = universe_key
            logger.info("step %d: rebuilt forecast recipe (full fit)", step)
        else:
            assert self._recipe is not None
            fit = recondition(self._recipe, data, prob, self.pipeline_config)
            logger.info("step %d: reconditioned forecast (cached recipe)", step)

        forecasts = self._forecast(panel, universe, fit, data, seed)
        self._cached_inputs = self._build_inputs(forecasts)
        return self._cached_inputs

    def _forecast(
        self,
        panel,
        universe: AssetUniverse,
        fit: FittedUniverse,
        data,
        seed: int | None,
    ) -> ForecastPaths:
        return _forecast_from_fit(
            panel=panel,
            asset_universe=universe,
            universe_fit=fit,
            last_data=data.tail(1),
            n_rows=data.height,
            seed=seed,
            simulation_config=self.simulation_config,
        )

    def _build_inputs(self, forecasts: ForecastPaths) -> PolicyInputs:
        cfg = self.input_config
        return PolicyInputs.from_policy_sources(
            forecasts=forecasts,
            cash_path=cfg.cash_path,
            expected_returns=cfg.expected_returns,
            risk=cfg.risk,
            horizons=cfg.horizons,
            subset=cfg.subset,
            pnl_type=cfg.pnl_type,
            expectation_tolerance=cfg.expectation_tolerance,
            mean_decay=cfg.mean_decay,
            step_size=cfg.step_size,
            periods_per_year=cfg.periods_per_year,
        )

    def _seed_for(self, step: int) -> int | None:
        return None if self.seed is None else self.seed + step


class ForecastingMPOPolicy:
    """Batteries-included MPO strategy for a backtest.

    Bundles an :class:`MPOPolicy` optimizer with a :class:`BacktestForecaster`
    (forecast cadence + recipe cache). It exposes exactly what ``run_backtest``
    needs — ``policy_inputs_at`` (produce PolicyInputs for a rebalance) and
    ``decide`` (optimize against them) — so ``run_backtest(market, policy)``
    works with no separate ``forecaster=`` argument. ``construction`` stays a
    pure optimizer; forecast-then-optimize orchestration lives here.
    """

    def __init__(
        self,
        policy: MPOPolicy,
        input_config: PolicyInputConfig,
        *,
        recipe_every: int = 12,
        forecast_every: int = 1,
        pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
        simulation_config: SimulationForecastConfig = DEFAULT_SIMULATION_CONFIG,
        seed: int | None = None,
        min_history: int = 0,
        name: str | None = None,
    ) -> None:
        self._policy = policy
        self._forecaster = BacktestForecaster(
            input_config,
            recipe_every=recipe_every,
            forecast_every=forecast_every,
            pipeline_config=pipeline_config,
            simulation_config=simulation_config,
            seed=seed,
        )
        self.name = name or policy.name
        self.min_history = min_history

    def policy_inputs_at(self, snapshot: MarketSnapshot, step: int) -> PolicyInputs:
        return self._forecaster.policy_inputs_at(snapshot, step)

    def decide(
        self,
        state: PortfolioState,
        policy_inputs: PolicyInputs | None = None,
        *,
        inputs: dict | None = None,
    ) -> PolicyDecision:
        if policy_inputs is None:
            raise ValueError(
                "ForecastingMPOPolicy.decide needs PolicyInputs (run_backtest "
                "supplies them via policy_inputs_at); use decide_at() for a "
                "one-shot decision from a snapshot."
            )
        return self._policy.optimize(
            state=state, policy_inputs=policy_inputs, inputs=inputs
        )

    def decide_at(
        self, snapshot: MarketSnapshot, state: PortfolioState, step: int = 0
    ) -> PolicyDecision:
        """One-shot forecast + optimize from a snapshot (outside run_backtest)."""
        return self.decide(state, self.policy_inputs_at(snapshot, step))
