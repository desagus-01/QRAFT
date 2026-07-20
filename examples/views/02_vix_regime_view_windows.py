# %%
"""STRESS_INDEX-regime view windows with diagnostics and plotting."""

import numpy as np

from qraft import Views
from qraft.core.scenarios.view_types import MeanView, QuantileView, RankingView
from qraft.core.scenarios.views import ViewWindow
from qraft.utils.example_data import synthetic_vix_market

# %%
# Build a synthetic market with three broad STRESS_INDEX regimes.
market_without_views = synthetic_vix_market()

# %%
# Pick non-overlapping windows. Views are active only inside these intervals.
bars = market_without_views.trading_bars
risk_on_start, risk_on_end = bars[120], bars[179]
risk_off_start, risk_off_end = bars[270], bars[359]
normalizing_start, normalizing_end = bars[450], bars[-1]

risk_on_returns = market_without_views.returns_through(risk_on_start)
risk_off_returns = market_without_views.returns_through(risk_off_start)
normalizing_returns = market_without_views.returns_through(normalizing_start)

low_stress_move = float(
    np.quantile(risk_on_returns.values.get_column("STRESS_INDEX").to_numpy(), 0.25)
)
high_stress_move = float(
    np.quantile(risk_off_returns.values.get_column("STRESS_INDEX").to_numpy(), 0.80)
)
normal_stress_move = float(
    np.quantile(normalizing_returns.values.get_column("STRESS_INDEX").to_numpy(), 0.50)
)

# %%
# Define regime views from STRESS_INDEX state assumptions plus cross-asset ranking views.
view_windows = [
    ViewWindow(
        risk_on_start,
        risk_on_end,
        Views(
            [
                MeanView("STRESS_INDEX", "<=", low_stress_move),
                RankingView(["NOVA", "LUMEN", "CYPHER", "TERRA", "ORION"]),
            ],
            confidence=0.60,
        ),
        name="risk_on_low_stress",
    ),
    ViewWindow(
        risk_off_start,
        risk_off_end,
        Views(
            [
                MeanView("STRESS_INDEX", ">=", high_stress_move),
                RankingView(["ORION", "TERRA", "CYPHER", "LUMEN", "NOVA"]),
            ],
            confidence=0.75,
        ),
        name="risk_off_high_stress",
    ),
    ViewWindow(
        normalizing_start,
        normalizing_end,
        Views(
            [
                MeanView("STRESS_INDEX", "==", normal_stress_move),
                QuantileView("NOVA", 0.10, 0.07),
            ],
            confidence=0.70,
        ),
        name="normalizing_stress",
    ),
]

market = market_without_views.with_views(view_windows)

# %%
# Retrieve the viewed distributions for all registered windows.
view_names = [
    report.name for report in market.viewed_returns() if report.name is not None
]

# %%
# Plot each registered view window.
for view_name in view_names:
    market.plot_view(view_name, prob_mode="regular")
