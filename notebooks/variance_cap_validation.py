# %%
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from qraft import AssetUniverse, CMAConfig, LogConfig, PipelineConfig, setup_logging
from qraft.backtest.market import MarketData, WindowWeighting
from qraft.core.configs import SimulationForecastConfig
from qraft.forecast.pipelines.fitted_universe import apply_forecast_recipe
from qraft.forecast.pipelines.forecasting import draw_innovations
from qraft.forecast.run import build_forecast_recipe_history
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.WARN))


# %%
# Configuration: keep this moderate for interactive runs, then scale up once it works.
DATA_PATH = "./data/tiingo_sample.csv"
FACTORS_PATH = "./data/tiingo_factors.csv"
CASH_PATH = "data/cash.csv"

MIN_PRICE = 10
ASSET_SLICE = slice(10, 90)
MIN_HISTORY = 150
REFIT_FRACTION = 5
MAX_PERIODS: int | None = 2
SEED = 4

SIM_CONFIG = SimulationForecastConfig(
    horizon=15,
    n_sims=20_000,
    method="cma",
    back_to_price=False,
    cma_config=CMAConfig(target_copula="t"),
)

LEGACY_CAP_FACTOR = 4.0
OUTPUT_CSV: str | None = "notebooks/variance_cap_validation_results.csv"


# %%
@dataclass(frozen=True, slots=True)
class FitCapValidationRow:
    period_index: int
    period_start: object
    period_end: object
    asset: str
    legacy_cap: float | None
    bind_count: int
    step_count: int
    bind_rate: float
    capped_annualized_vol: float
    uncapped_annualized_vol: float
    annualized_vol_delta: float
    annualized_vol_delta_pct: float
    capped_cvar_5: float
    uncapped_cvar_5: float
    cvar_5_delta: float
    cvar_5_delta_pct: float


def annualized_vol(flat_returns: np.ndarray, periods_per_year: float) -> float:
    if flat_returns.size < 2:
        return 0.0
    return float(np.std(flat_returns, ddof=1) * np.sqrt(periods_per_year))


def cvar_5_loss(flat_returns: np.ndarray) -> float:
    if flat_returns.size == 0:
        return 0.0
    losses = -np.asarray(flat_returns, dtype=float)
    cutoff = np.quantile(losses, 0.95)
    tail = losses[losses >= cutoff]
    return float(np.mean(tail)) if tail.size else float(cutoff)


def set_legacy_caps(fit) -> dict[str, float | None]:
    original_caps: dict[str, float | None] = {}
    for asset, forecast in fit.simulation_forecasts.items():
        original_caps[asset] = forecast.state0.var_cap
        vol_res = fit.models[asset].volatility_res
        if vol_res is None:
            forecast.state0.var_cap = None
            continue
        sigma2 = np.asarray(vol_res.conditional_volatility, dtype=float) ** 2
        forecast.state0.var_cap = float(LEGACY_CAP_FACTOR * np.max(sigma2))
    return original_caps


def restore_caps(fit, caps: dict[str, float | None]) -> None:
    for asset, cap in caps.items():
        fit.simulation_forecasts[asset].state0.var_cap = cap


def pct_delta(delta: float, base: float) -> float:
    return float(delta / abs(base)) if np.isfinite(base) and abs(base) > 0 else 0.0


def validate_period(
    *,
    period_index: int,
    period,
    market: MarketData,
    pipeline_config: PipelineConfig,
    simulation_config: SimulationForecastConfig,
    seed: int,
) -> list[FitCapValidationRow]:
    panel = market.history_through(period.start)
    data = panel.to_frame()
    fit = apply_forecast_recipe(period.recipe, data, panel.prob, pipeline_config)

    innovations = draw_innovations(
        invariants=fit.invariants,
        horizon=simulation_config.horizon,
        n_sims=simulation_config.n_sims,
        seed=seed,
        method=simulation_config.method,
        cma_config=simulation_config.cma_config,
    )

    original_caps = set_legacy_caps(fit)
    capped = fit.simulate(innovations.values)
    diagnostics = {
        asset: dict(forecast.variance_cap_diagnostics)
        for asset, forecast in fit.simulation_forecasts.items()
    }

    for forecast in fit.simulation_forecasts.values():
        forecast.state0.var_cap = None
        forecast.variance_cap_diagnostics = {}
    uncapped = fit.simulate(innovations.values)
    restore_caps(fit, original_caps)

    rows: list[FitCapValidationRow] = []
    for asset in fit.assets:
        capped_returns = np.asarray(capped[asset], dtype=float).ravel()
        uncapped_returns = np.asarray(uncapped[asset], dtype=float).ravel()

        capped_vol = annualized_vol(capped_returns, market.config.periods_per_year)
        uncapped_vol = annualized_vol(uncapped_returns, market.config.periods_per_year)
        vol_delta = uncapped_vol - capped_vol

        capped_cvar = cvar_5_loss(capped_returns)
        uncapped_cvar = cvar_5_loss(uncapped_returns)
        cvar_delta = uncapped_cvar - capped_cvar

        diag = diagnostics.get(asset, {})
        rows.append(
            FitCapValidationRow(
                period_index=period_index,
                period_start=period.start,
                period_end=period.end,
                asset=asset,
                legacy_cap=diag.get("cap_value"),
                bind_count=int(diag.get("bind_count") or 0),
                step_count=int(diag.get("step_count") or 0),
                bind_rate=float(diag.get("bind_rate") or 0.0),
                capped_annualized_vol=capped_vol,
                uncapped_annualized_vol=uncapped_vol,
                annualized_vol_delta=vol_delta,
                annualized_vol_delta_pct=pct_delta(vol_delta, capped_vol),
                capped_cvar_5=capped_cvar,
                uncapped_cvar_5=uncapped_cvar,
                cvar_5_delta=cvar_delta,
                cvar_5_delta_pct=pct_delta(cvar_delta, capped_cvar),
            )
        )
    return rows


# %%
data, factors_cols = import_tickers_and_factors(DATA_PATH, FACTORS_PATH)
cash = pl.read_csv(CASH_PATH, try_parse_dates=True)

cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= np.log(MIN_PRICE)  # type: ignore[arg-type]
    )
]
data = data.select(cols_to_keep)

tradable_assets = list(data.columns[ASSET_SLICE])
universe = AssetUniverse(assets=tradable_assets, factors=list(factors_cols))
data = data.select("date", *universe.all_tickers)

market = MarketData.from_log_prices(
    data,
    universe,
    cash=cash,
    weighting=WindowWeighting("state_smooth", half_life=60),
)

pipeline_config = PipelineConfig(exclude_non_invariants=False)


# %%
recipe_history = build_forecast_recipe_history(
    market,
    min_history=MIN_HISTORY,
    refit_every=max(1, int(data.height / REFIT_FRACTION)),
    seed=SEED,
    pipeline_config=pipeline_config,
)


# %%
rows: list[FitCapValidationRow] = []
periods_to_validate = recipe_history.periods[:MAX_PERIODS]
for period_index, period in enumerate(periods_to_validate):
    rows.extend(
        validate_period(
            period_index=period_index,
            period=period,
            market=market,
            pipeline_config=pipeline_config,
            simulation_config=SIM_CONFIG,
            seed=SEED + period_index,
        )
    )

results = pl.DataFrame([asdict(row) for row in rows])
results


# %%
summary = results.select(
    pl.len().alias("asset_periods"),
    (pl.col("bind_rate") > 0).mean().alias("share_asset_periods_bound"),
    pl.col("bind_rate").mean().alias("mean_bind_rate"),
    pl.col("bind_rate").max().alias("max_bind_rate"),
    pl.col("annualized_vol_delta").mean().alias("mean_vol_delta"),
    pl.col("annualized_vol_delta_pct").mean().alias("mean_vol_delta_pct"),
    pl.col("cvar_5_delta").mean().alias("mean_cvar_5_delta"),
    pl.col("cvar_5_delta_pct").mean().alias("mean_cvar_5_delta_pct"),
)
summary


# %%
worst_binding = results.sort("bind_rate", descending=True).head(20)
worst_risk_delta = results.sort("cvar_5_delta_pct", descending=True).head(20)

print("Summary")
print(summary)
print("\nWorst binding")
print(worst_binding)
print("\nLargest uncapped-vs-capped CVaR increases")
print(worst_risk_delta)

if OUTPUT_CSV is not None:
    results.write_csv(OUTPUT_CSV)
    print(f"\nWrote detailed results to {OUTPUT_CSV}")
