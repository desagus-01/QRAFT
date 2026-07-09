import numpy as np
from numpy.typing import NDArray

from qraft.construction.policies import PolicyDecision


def target_weights_full(
    decision: PolicyDecision, asset_order: list[str]
) -> NDArray[np.floating]:
    idx = {asset: i for i, asset in enumerate(asset_order)}
    weights = np.zeros(len(asset_order))
    for asset, weight in zip(decision.asset_order, decision.target_weights_risk):
        weights[idx[asset]] = weight  # assets absent from the decision stay 0
    return weights
