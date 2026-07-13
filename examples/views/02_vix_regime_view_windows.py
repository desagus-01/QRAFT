# %%
"""VIX-regime view windows with diagnostics and plotting."""

import numpy as np

from qraft import Views
from qraft.core.scenarios.view_types import MeanView, QuantileView, RankingView
from qraft.core.scenarios.views import ViewWindow
from qraft.utils.example_data import synthetic_vix_market

# %%
# Build a synthetic market with three broad VIX regimes.
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

low_vix_move = float(
    np.quantile(risk_on_returns.values.get_column("VIX").to_numpy(), 0.25)
)
high_vix_move = float(
    np.quantile(risk_off_returns.values.get_column("VIX").to_numpy(), 0.80)
)
normal_vix_move = float(
    np.quantile(normalizing_returns.values.get_column("VIX").to_numpy(), 0.50)
)

# %%
# Define regime views from VIX state assumptions plus cross-asset ranking views.
view_windows = [
    ViewWindow(
        risk_on_start,
        risk_on_end,
        Views(
            [
                MeanView("VIX", "<=", low_vix_move),
                RankingView(["SPY", "GLD", "TLT"]),
            ],
            confidence=0.60,
        ),
        name="risk_on_low_vix",
    ),
    ViewWindow(
        risk_off_start,
        risk_off_end,
        Views(
            [
                MeanView("VIX", ">=", high_vix_move),
                RankingView(["TLT", "GLD", "SPY"]),
            ],
            confidence=0.75,
        ),
        name="risk_off_high_vix",
    ),
    ViewWindow(
        normalizing_start,
        normalizing_end,
        Views(
            [
                MeanView("VIX", "==", normal_vix_move),
                QuantileView("SPY", 0.10, 0.07),
            ],
            confidence=0.70,
        ),
        name="normalizing_vix",
    ),
]

market = market_without_views.with_views(view_windows)

# %%
# Retrieve the viewed distributions for all registered windows.
view_names = [
    report.name for report in market.viewed_returns() if report.name is not None
]
print(view_names)

# %%
# Plot each registered view window.
for view_name in view_names:
    market.plot_view(view_name, prob_mode="regular")
