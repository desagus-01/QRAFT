# Contributing

## Docstring Style

QRAFT uses reST-lite prose, with NumPy sections only when they add information. The summary line is one sentence, ends with a period, and should fit on one line. Add a body only to explain semantics, invariants, units, shapes, or conventions that are not obvious from the signature. Do not restate parameter or return types that already appear in annotations.

Use inline literals with double backticks, such as ``weights`` or ``tail="left"``. Use Sphinx roles such as `:class:` and `:meth:` sparingly, only when the cross-reference is materially useful.

Use type-less NumPy ``Parameters`` and ``Returns`` blocks only for numeric functions where shape, units, or tail convention matters. Treat ``risk_attribution.py`` as the exemplar for these cases.

For public dataclass configs, describe the key knobs in class-docstring prose. Keep ``#`` field comments for lower-level detail that helps readers understand individual fields.

Facade methods should name the returned object and the next call the user is expected to make.

Example facade method:

```python
def walk_forward(self) -> WalkForwardReport:
    """Return a ``WalkForwardReport``; call ``plot_walk_forward_report`` to inspect it."""
```

Example numeric function:

```python
def tail_mean(losses: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """Return the probability-weighted mean loss in the left tail.

    Parameters
    ----------
    losses:
        Scenario losses with shape ``(n_scenarios,)``.
    probs:
        Scenario probabilities that sum to one.
    alpha:
        Tail mass in probability units.

    Returns
    -------
    Mean loss over the worst ``alpha`` probability mass.
    """
```
