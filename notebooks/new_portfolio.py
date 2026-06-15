# %%
import logging

import numpy as np

from qraft import (
    AssetUniverse,
    LogConfig,
    run_forecast,
    setup_logging,
)
from qraft.core import (
    ScenarioPanel,
    state_smooth_probs,
)
from qraft.core.configs import SimulationForecastConfig
from qraft.helper import profile
from qraft.utils.tiingo import import_tickers_and_factors

logging.getLogger("py.warnings").setLevel(logging.ERROR)
setup_logging(LogConfig(level=logging.INFO))

# %%
# ── Data loading ─────────────────────────────────────────────────────
data, factors_cols = import_tickers_and_factors(
    "./data/tiingo_sample.csv",
    "./data/tiingo_factors.csv",
)

min_price = 9


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

tradable_assets = list(data.columns[10:90])
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
#
# posterior_panel = Views(
#     [
#         MeanView("DHIL", ">=", 0.002),
#         CorrView(("MTX", "CUBE"), ">=", 0.75),
#         RankingView(["DHIL", "MTX", "NBIX"]),
#     ],
#     confidence=0.8,
# ).apply(original_panel)
#

# %%
# ── Forecasting ──────────────────────────────────────────────────────
with profile():
    forecasts = run_forecast(
        panel=posterior_panel,
        seed=3,
        universe=universe,
        simulation_config=SimulationForecastConfig(n_sims=10_000),
    )

# forecasts.plot_asset_paths()
# # %%
# constraints: list[PortfolioConstraint] = [
#     LongOnly(),
#     FullyInvested(constraint_type="soft", soft_weight=1.0),
#     # FullyInvested(),
#     MinCashWeight(limit=0.4, constraint_type="soft", soft_weight=1.0),
#     # MaxWeight(limit=0.09),
#     # MaxWeightTopN(top_n=10, sum_limit=0.4, constraint_type="soft", soft_weight=500),
#     TurnoverLimit(limit=0.80),
# ]
#
# policy = MPOPolicy.preset(
#     objective_type="cvar_cuts",
#     risk_aversion=0.1,
#     cash_path="data/cash.csv",
#     constraints=constraints,
#     expectation_tolerance=0.1,
# )
#
# # %%
# rng = np.random.default_rng(seed=1)
# rand_shares = rng.integers(low=10, high=75, size=len(universe.assets))
# state = PortfolioState.from_forecast_and_assets(
#     asset_forecasts=forecasts,
#     assets=universe.assets,
#     shares=rand_shares,
#     cash=100_000,
# )
#
# # %%
# projection = policy.decide(state, forecasts)
# projection.plot(type="cum_performance")
# x = projection.risk(forecasts)
# x.effective_bets().plot()
# x.risk_contribution("cvar")
