from typing import Annotated, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator

Sign: TypeAlias = Literal["<=", ">=", "=="]


def validate_prob_vector(a: NDArray[np.float64]) -> NDArray[np.float64]:
    if a.ndim != 1:
        raise ValueError("Array must be 1D.")
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        raise ValueError("Array must not contain NaN or infinite values.")
    if np.any(a < 0):
        raise ValueError("All probabilities must be non-negative.")
    if not np.isclose(a.sum(dtype=np.float64), 1.0, rtol=0, atol=1e-5):
        raise ValueError(
            f"Probabilities must sum to 1. Currently this is {a.sum(dtype=np.float64)}"
        )
    return a


ProbVector = Annotated[NDArray[np.float64], AfterValidator(validate_prob_vector)]
