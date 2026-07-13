# %%
"""Latest VIX-state view with diagnostics and plotting."""

import numpy as np

from qraft import Views
from qraft.core.scenarios.view_types import MeanView, RankingView
from qraft.core.scenarios.views import ViewWindow
from qraft.utils.example_data import synthetic_vix_market

# %%
# Build a synthetic market where VIX is a non-tradable state indicator.
market_without_views = synthetic_vix_market()

# %%
# Express the latest market view as a high-VIX state.
as_of = market_without_views.trading_bars[-1]
returns = market_without_views.returns_through(as_of)
high_vix_move = float(np.quantile(returns.values.get_column("VIX").to_numpy(), 0.80))

latest_view = ViewWindow(
    start=as_of,
    end=as_of,
    views=Views(
        [
            MeanView("VIX", ">=", high_vix_move),
            RankingView(["TLT", "GLD", "SPY"]),
        ],
        confidence=1.0,
    ),
    name="latest_high_vix_state",
)

market = market_without_views.with_views([latest_view])

# %%
# Pull the active view report for the latest date.
report = market.view_report(as_of)

# Diagnostics tell you whether the posterior is numerically healthy and which
# constraints are binding or influential.
print(report.diagnostics)

# Plot where probability mass moved through the historical scenario set.
report.plot()
