# %%
import numpy as np

from qraft import Views
from qraft.core.scenarios.view_types import MeanView, RankingView
from qraft.core.scenarios.views import ViewWindow
from qraft.utils.example_data import synthetic_market_example

# %%
# Build a synthetic market where STRESS_INDEX, GROWTH_PULSE, and RATE_WAVE are non-tradable state indicators.
market_without_views = synthetic_market_example()

# %%
# Express the latest market view as a high-stress state.
as_of = market_without_views.trading_bars[-1]
returns = market_without_views.returns_through(as_of)
high_stress_move = float(
    np.quantile(returns.values.get_column("STRESS_INDEX").to_numpy(), 0.80)
)

latest_view = ViewWindow(
    start=as_of,
    end=as_of,
    views=Views(
        [
            MeanView("STRESS_INDEX", ">=", high_stress_move),
            RankingView(["ORION", "TERRA", "CYPHER", "LUMEN", "NOVA"]),
        ],
        confidence=1.0,
    ),
    name="latest_high_stress_state",
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
