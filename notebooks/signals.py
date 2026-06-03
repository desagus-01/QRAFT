# %%
import logging

from forecast.config import LogConfig
from pipelines.forecasting import AssetUniverse
from probability.distributions import state_smooth_probs
from scenarios.panel import ScenarioPanel
from signals.raw_signals import ewma_signal
from signals.signal_processing import process_signal
from utils.log import setup_logging
from utils.tiingo import import_tickers_and_factors

setup_logging(LogConfig(level=logging.WARNING))

# %%
# ── Data loading ─────────────────────────────────────────────────────

data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

cols_to_keep = [
    col
    for col in data.columns
    if col == "date" or (data[col].null_count() == 0 and data[col].min() >= 1)
]

data = data.select(cols_to_keep)

tradable_assets = list(data.columns[15:25])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
assets_df = data.select("date", *universe.assets)

# %%
assets_df
prob_ex = state_smooth_probs(
    data.height,
    half_life=120,
    time_based=True,
)

asset_panel = ScenarioPanel.from_frame(
    assets_df,
    prob=prob_ex,
)
# %%

mom_sig = ewma_signal(
    asset_panel.values, asset_panel.dates, half_life="60d", drop_nulls=True
)


process_signal(mom_sig, 60, 60, "average", "winsorize")
