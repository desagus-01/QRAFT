"""Policy candidate overlay example.

Run this file from the repository root, or copy the pattern into a notebook.
It demonstrates Phase 2 only: expand a user-created policy into immutable
candidate policies. It does not run the Phase 3 backtest evaluator.
"""

from qraft.backtest.selection import expand_candidates
from qraft.construction import FullyInvested, LongOnly, MPOPolicy

policy = MPOPolicy.preset(
    "mean_covariance",
    risk_aversion=1.0,
    name="mean_covariance_base",
    constraints=(LongOnly(), FullyInvested()),
)

candidates = expand_candidates(
    policy,
    {
        "risk_aversion": [0.5, 1.0, 2.0],
        "transaction_cost_weight": [0.5, 1.0],
    },
)

for candidate in candidates:
    preset = candidate.policy.problem._preset
    assert preset is not None
    print(
        candidate.params,
        "->",
        candidate.policy.name,
        "risk_aversion=",
        preset.risk_aversion,
        "transaction_cost_weight=",
        preset.transaction_cost_weight,
    )

# Phase 3 will consume these candidates like:
#
# table = precompute_inputs(market, schedule, forecast_provider, policy.min_history)
# inputs = PrecomputedInputsProvider(table)
# for candidate in candidates:
#     result = run_backtest(market, candidate.policy, schedule=schedule, inputs=inputs)
