# %%
import logging

import numpy as np

from qraft import (
    AssetUniverse,
    CMAConfig,
    LogConfig,
    MPOPolicy,
    PolicyInputs,
    PolicyProjection,
    PortfolioState,
    run_forecast,
    setup_logging,
)
from qraft.construction import (
    FullyInvested,
    MinCashWeight,
    TurnoverLimit,
)
from qraft.construction.optimization.constraints import LongOnly, PortfolioConstraint
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
        include_fit_diagnostics=True,
        simulation_config=SimulationForecastConfig(
            method="cma", n_sims=10_000, cma_config=CMAConfig(target_copula="t")
        ),
    )


# %%
forecasts.plot_asset_paths()
# %%
constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(constraint_type="soft", soft_weight=1.0),
    MinCashWeight(limit=0.3, constraint_type="soft", soft_weight=1.0),
    TurnoverLimit(limit=0.80),
]

policy = MPOPolicy.preset(
    objective_type="cvar_auto",
    risk_aversion=2.0,
    alpha=0.05,
    constraints=constraints,
)

rng = np.random.default_rng(seed=1)
rand_shares = rng.integers(low=10, high=75, size=len(universe.assets))
state = PortfolioState.from_forecast_and_assets(
    asset_forecasts=forecasts,
    assets=universe.assets,
    shares=rand_shares,
    cash=100_000,
)

# ── Build PolicyInputs: historical expected returns + forecast risk ──
policy_inputs = PolicyInputs.from_policy_sources(
    forecasts=forecasts,
    cash_path="data/cash.csv",
    expected_returns="forecast",
    history=posterior_panel,
    risk="both",
    expectation_tolerance=0.2,
)

policy_inputs.mean_frame

# %%
decision = policy.optimize(state, policy_inputs)

# %%

decision.target_weights_risk

decision.target_cash_weight

PolicyProjection.from_decision(decision, forecasts, state=state).plot(type="value")

# %%
# ── Multi-period frontier sweep (historical mean + forecast scenarios) ──
# gammas = np.logspace(-2, 1, 12)
# frontier_points = []
# for gamma in gammas:
#     p = MPOPolicy.preset(
#         objective_type="cvar_auto",
#         risk_aversion=gamma,
#         alpha=0.05,
#         constraints=constraints,
#     )
#     try:
#         d = p.optimize(state, policy_inputs)
#     except RuntimeError:
#         continue
#
#     mpo_result = d.diagnostics
#     ante = ex_ante_metrics(mpo_result, policy_inputs)
#     cvar_val = in_model_cvar(mpo_result, policy_inputs, 0.05)
#     cvar_term = ex_post_terminal_cvar(mpo_result, policy_inputs, 0.05)
#
#     frontier_points.append(
#         {
#             "gamma": gamma,
#             "status": mpo_result.status,
#             "expected_return": ante["expected_return"],
#             "sum_horizon_volatility": ante["volatility"],
#             "turnover": mpo_result.turnover,
#             "cash_weight": d.target_cash_weight,
#             "objective_value": mpo_result.objective_value,
#             "cvar_in_model": cvar_val,
#             "cvar_terminal": cvar_term,
#         }
#     )
# # %%
# frontier_table = pl.DataFrame(frontier_points).sort(
#     "sum_horizon_volatility", descending=True
# )
# frontier_table
#
# # %%
