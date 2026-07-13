import numpy as np
import pytest

from qraft.risk.dimensionality_reduction import minimum_torsion_matrix


def test_minimum_torsion_approximate_returns_identity_for_independent_data() -> None:
    data = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    prob = np.full(4, 0.25)

    torsion = minimum_torsion_matrix(data, prob, method="approximate")

    assert np.allclose(torsion, np.eye(2), atol=1e-12)


def test_minimum_torsion_rejects_max_iter_for_approximate_method() -> None:
    with pytest.raises(ValueError, match="Method selected must be `exact`"):
        minimum_torsion_matrix(np.eye(3), np.full(3, 1 / 3), max_iter=10)


def test_minimum_torsion_exact_accepts_default_max_iter() -> None:
    data = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    torsion = minimum_torsion_matrix(data, np.full(4, 0.25), method="exact")

    assert torsion.shape == (2, 2)


def test_minimum_torsion_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        minimum_torsion_matrix(
            np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            np.full(4, 0.25),
            method="bad",  # type: ignore[arg-type]
        )
