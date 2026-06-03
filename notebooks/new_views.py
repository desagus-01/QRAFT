# %%
import logging

import numpy as np
from forecast.config import LogConfig
from forecast.pipelines.forecasting import AssetUniverse
from forecast.probability.distributions import state_smooth_probs
from forecast.scenarios.panel import ScenarioPanel
from forecast.scenarios.transforms import CMA, Views, apply_scenario_transforms
from forecast.scenarios.types import CorrView, MeanView, RankingView
from utils.log import setup_logging
from utils.tiingo import import_tickers_and_factors

setup_logging(LogConfig(level=logging.WARNING))

# %%
# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

min_price = 15

cols_to_keep = [
    col
    for col in data.columns
    if col == "date"
    or (
        data[col].null_count() == 0
        and data[col].dtype.is_numeric()
        and float(data[col].min()) >= np.log(min_price)  # type: ignore[arg-type]
    )
]

data = data.select(cols_to_keep)

tradable_assets = list(data.columns[10:30])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)
# %%
forecast_horizon = 20
n_sims = 30_000

prob_ex = state_smooth_probs(
    data.height,
    half_life=data.height / 2,
    time_based=True,
)

historical_panel = ScenarioPanel.from_frame(
    data,
    prob=prob_ex,
)
# %%
historical_panel
# %%
views = Views(
    [
        MeanView("DHIL", ">=", 0.002),
        CorrView(("MTX", "CUBE"), ">=", 0.75),
        RankingView(["DHIL", "MTX", "NBIX"]),
    ],
    confidence=0.8,
)

cma = CMA(target_copula="t", target_marginals=None, seed=1)

posterior_panel = apply_scenario_transforms(historical_panel, [views, cma])

# %%

posterior_panel
