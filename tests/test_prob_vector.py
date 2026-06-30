import numpy as np
import pytest

from qraft.core.probability.prob_vector import as_prob_vector


def test_as_prob_vector_validates_and_freezes() -> None:
    prob = as_prob_vector([0.25, 0.75], length=2)

    np.testing.assert_allclose(prob, [0.25, 0.75])
    assert prob.dtype == np.float64
    assert not prob.flags.writeable


def test_as_prob_vector_copies_by_default() -> None:
    source = np.array([0.25, 0.75])

    prob = as_prob_vector(source)
    source[0] = 1.0

    np.testing.assert_allclose(prob, [0.25, 0.75])


@pytest.mark.parametrize(
    "values, match",
    [
        (np.array([[0.5, 0.5]]), "1D"),
        (np.array([0.5, np.nan]), "NaN"),
        (np.array([1.1, -0.1]), "non-negative"),
        (np.array([0.2, 0.2]), "sum to 1"),
    ],
)
def test_as_prob_vector_rejects_invalid_probabilities(values, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        as_prob_vector(values)


def test_as_prob_vector_validates_expected_length() -> None:
    with pytest.raises(ValueError, match="expected length 3"):
        as_prob_vector([0.25, 0.75], length=3)
