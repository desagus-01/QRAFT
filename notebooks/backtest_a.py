# %%
import logging

import numpy as np

from qraft import (
    AssetUniverse,
    LogConfig,
    setup_logging,
)
from qraft.core import (
    ScenarioPanel,
    state_smooth_probs,
)
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.INFO))

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

tradable_assets = list(data.columns[10:80])
factors_cols = list(factors_cols)
universe = AssetUniverse(assets=tradable_assets, factors=factors_cols)
data = data.select("date", *universe.all_tickers)

# %%
# ── Build historical ScenarioPanel ───────────────────────────────────

prob_ex = state_smooth_probs(
    data.height,
    half_life=data.height / 2,
    time_based=True,
)

posterior_panel = ScenarioPanel.from_log_prices(
    data,
    prob=prob_ex,
)
# %%

posterior_panel.to_frame()
