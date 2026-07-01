import numpy as np
import polars as pl

from qraft.core.probability.prob_vector import as_prob_vector
from qraft.utils.helpers import indicator_quantile_marginal


def test_indicator_quantile_marginal_uses_weighted_threshold() -> None:
    data = pl.DataFrame({"A": [0.0, 1.0, 2.0]})
    prob = as_prob_vector(np.array([0.8, 0.1, 0.1]))

    indicator = indicator_quantile_marginal(data, 0.5, prob=prob)

    assert indicator.get_column("quant_ind").to_list() == [1, 0, 0]
