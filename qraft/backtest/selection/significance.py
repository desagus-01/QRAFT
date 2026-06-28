from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from qraft.core import metrics

_EULER = 0.5772156649015329


def _moments(returns: NDArray[np.floating]) -> tuple[float, float, float, int]:
    """Per-period Sharpe, skew, non-excess kurtosis, and sample length."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = returns.size
    if n < 2:
        return float("nan"), 0.0, 3.0, n
    if returns.std(ddof=1) == 0:
        return 0.0, 0.0, 3.0, n

    sharpe = metrics.per_period_sharpe(returns)
    centred = returns - returns.mean()
    pop_std = returns.std(ddof=0)
    skew = float(np.mean(centred**3) / pop_std**3)
    kurt = float(np.mean(centred**4) / pop_std**4)
    return sharpe, skew, kurt, n


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    """SR0: expected maximum per-period Sharpe under the null of no skill.

    Bailey & Lopez de Prado (2014): grows with the number of trials and their
    Sharpe dispersion, so a higher SR0 is a higher bar the strategy must clear.
    """
    sr = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    n = sr.size
    var = float(np.var(sr, ddof=1))
    if var <= 0:
        raise ValueError("All trial sharpes are the same, variance is zero.")
    z1 = float(norm.ppf(1.0 - 1.0 / n))
    z2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    return math.sqrt(var) * ((1.0 - _EULER) * z1 + _EULER * z2)


def deflated_sharpe(
    returns: NDArray[np.floating],
    trial_sharpes: Sequence[float],
) -> float:
    """Probability the selected strategy's true Sharpe is positive after
    correcting for the number of trials and non-normality.
    """
    sharpe, skew, kurt, n = _moments(returns)
    sr0 = expected_max_sharpe(trial_sharpes)
    variance = max(1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2, 1e-12)
    z = (sharpe - sr0) * math.sqrt(n - 1) / math.sqrt(variance)
    return float(norm.cdf(z))


def prob_of_backtest_overfit(performance: NDArray[np.floating]) -> float:
    """Probability of Backtest Overfitting via Combinatorial Symmetric CV.

    ``performance`` is an ``(n_blocks, n_candidates)`` matrix of per-block
    performance (e.g. mean return). For every way to split the blocks into an
    in-sample / out-of-sample half, take the in-sample-best candidate and check
    whether it lands at or below the OOS median; PBO is that fraction. Raises
    ValueError when the matrix is not 2D, too small, or has an odd number of blocks.
    """
    if performance.ndim != 2:
        raise ValueError(f"performance must be a 2D array, got ndim={performance.ndim}")
    n_blocks, n_candidates = performance.shape
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(
            f"performance must have at least 2 blocks and an even number of blocks, "
            f"got n_blocks={n_blocks}"
        )
    if n_candidates < 2:
        raise ValueError(
            f"performance must have at least 2 candidates, got n_candidates={n_candidates}"
        )

    half = n_blocks // 2
    blocks = range(n_blocks)
    below = 0
    total = 0
    for is_blocks in combinations(blocks, half):
        is_set = set(is_blocks)
        oos_blocks = [b for b in blocks if b not in is_set]
        is_best = int(np.argmax(performance[list(is_blocks)].mean(axis=0)))
        oos_perf = performance[oos_blocks].mean(axis=0)
        oos_rank = int(np.argsort(np.argsort(oos_perf))[is_best])
        rel_rank = (oos_rank + 1) / (n_candidates + 1)
        below += rel_rank <= 0.5
        total += 1
    return float(below / total) if total else float("nan")
