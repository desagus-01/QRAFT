from qraft import MPOPolicy
from qraft.backtest.selection import expand_candidates
from qraft.construction import (
    FullyInvested,
    LongOnly,
    MinCashWeight,
    PortfolioConstraint,
    TurnoverLimit,
)

constraints: list[PortfolioConstraint] = [
    LongOnly(),
    FullyInvested(constraint_type="soft", soft_weight=1.0),
    MinCashWeight(limit=0.25),
    TurnoverLimit(limit=0.10),
]


policy = MPOPolicy.preset(
    objective_type="cvar_cuts",
    risk_aversion=0.01,
    constraints=constraints,
    min_history=504,
)


candidates = expand_candidates(
    policy,
    {
        "risk_aversion": [0.5, 1.0, 2.0],
        "transaction_cost_weight": [0.5, 1.0],
    },
)


candidates
# %%


# Phase 3 will consume these candidates like:
#
# table = precompute_inputs(market, schedule, forecast_provider, policy.min_history)
# inputs = PrecomputedInputsProvider(table)
# for candidate in candidates:
#     result = run_backtest(market, candidate.policy, schedule=schedule, inputs=inputs)
