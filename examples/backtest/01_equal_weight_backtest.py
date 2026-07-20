# %%
"""Backtest quickstart with an equal-weight policy."""

import logging

from qraft import (
    Backtest,
    BacktestConfig,
    EqualWeightPolicy,
    RebalanceSchedule,
)
from qraft.utils import LogConfig, setup_logging
from qraft.utils.example_data import synthetic_market_example

setup_logging(LogConfig(level=logging.INFO))

# %%
# Build a small synthetic market with five tradable assets and STRESS_INDEX, GROWTH_PULSE, and RATE_WAVE as factors.
market = synthetic_market_example()

# %%
# EqualWeightPolicy needs no forecasts or optimizer inputs. It is a useful first
# backtest because the only moving parts are the market, policy, and schedule.
policy = EqualWeightPolicy(target_cash_weight=0.05)

config = BacktestConfig(
    schedule=RebalanceSchedule("month_end"),
    initial_cash=100,
)

result = Backtest(
    market=market,
    policy=policy,
    config=config,
).run()

# %%
# BacktestResult stores the NAV path, rebalance periods, warnings, costs, and
# summary metrics. summary_df is convenient in notebooks and reports.
print(result.summary_df())
print(
    {
        "rebalances": len(result.periods),
        "final_nav": result.nav[-1],
        "total_cost": result.total_cost,
        "warnings": len(result.warnings_log),
    }
)

# %%
# Inspect realised rebalance activity.
print(result.period_execution_bars[:5])
print(result.period_turnovers[:5])

# %%
result.plot_nav()
result.plot_drawdown()
result.plot_turnover_and_costs()
