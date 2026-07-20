import math
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import norm

from qraft.backtest.selection.significance import (
    deflated_sharpe,
    expected_max_sharpe,
    prob_of_backtest_overfit,
)


def test_expected_max_sharpe_matches_published_formula() -> None:
    trial_sharpes = np.array([-0.1, 0.05, 0.2, 0.35, 0.5])
    euler = 0.5772156649015329
    n = trial_sharpes.size
    expected = np.std(trial_sharpes, ddof=1) * (
        (1 - euler) * norm.ppf(1 - 1 / n) + euler * norm.ppf(1 - 1 / (n * math.e))
    )

    assert expected_max_sharpe(trial_sharpes) == pytest.approx(expected)


def test_deflated_sharpe_matches_published_formula() -> None:
    returns = np.array([-0.03, -0.01, 0.005, 0.012, 0.02, 0.04, 0.055])
    trial_sharpes = np.array([-0.1, 0.05, 0.2, 0.35, 0.5])
    sharpe = returns.mean() / returns.std(ddof=1)
    centred = returns - returns.mean()
    pop_std = returns.std(ddof=0)
    skew = np.mean(centred**3) / pop_std**3
    kurt = np.mean(centred**4) / pop_std**4
    sr0 = expected_max_sharpe(trial_sharpes)
    denominator = math.sqrt(1 - skew * sharpe + ((kurt - 1) / 4) * sharpe**2)
    expected = norm.cdf((sharpe - sr0) * math.sqrt(len(returns) - 1) / denominator)

    assert deflated_sharpe(returns, trial_sharpes) == pytest.approx(expected)


def test_pbo_matches_independent_cscv_enumeration() -> None:
    performance = np.array(
        [
            [0.09, 0.03, -0.01],
            [0.07, 0.02, 0.04],
            [-0.08, 0.05, 0.01],
            [-0.06, 0.04, 0.03],
        ]
    )
    failures = []
    for in_sample in combinations(range(performance.shape[0]), 2):
        out_sample = sorted(set(range(performance.shape[0])) - set(in_sample))
        selected = int(np.argmax(performance[list(in_sample)].mean(axis=0)))
        ranks = np.argsort(np.argsort(performance[out_sample].mean(axis=0)))
        failures.append((ranks[selected] + 1) / (performance.shape[1] + 1) <= 0.5)

    assert prob_of_backtest_overfit(performance) == pytest.approx(np.mean(failures))
