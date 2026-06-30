from numpy.typing import NDArray
import numpy as np
from pydantic import validate_call

from qraft.core.probability.prob_vector import ProbVector, as_prob_vector
from qraft.globals import model_cfg


def kernel_smoothing(
    n: int,
    half_life: float,
    kernel_type: int,
    reference: float | None,
    time_based: bool = False,
) -> NDArray[np.float64]:
    if kernel_type <= 0:
        raise ValueError("kernel_type must be a positive integer")
    if n <= 0:
        raise ValueError("n must be positive")
    if half_life <= 0:
        raise ValueError("half_life must be positive")

    data = np.arange(n, dtype=np.float64)

    if time_based:
        dist_to_ref = (n - 1) - data  # distance from most recent point
    elif reference is not None:
        dist_to_ref = np.abs(data - reference)
    else:
        raise ValueError("reference must be provided when time_based=False")

    bandwidth = half_life / (np.log(2.0) ** (1.0 / kernel_type))
    return np.exp(-((dist_to_ref / bandwidth) ** kernel_type))


@validate_call(config=model_cfg, validate_return=True)
def uniform_probs(len: int) -> ProbVector:
    """
    Returns a uniform probability vector
    """
    return as_prob_vector(np.ones(len) / len)


@validate_call(config=model_cfg, validate_return=True)
def state_crisp_probs(length: int, condition_vector: NDArray[np.bool_]) -> ProbVector:
    """
    Returns a probability length based on state condition passed.
    """

    if length != len(condition_vector):
        raise ValueError("Input and condition length must have the same length.")

    p = np.zeros(len(condition_vector), dtype=np.float64)
    selected_indices = np.where(condition_vector)[0]

    p[selected_indices] = 1.0 / len(selected_indices)
    return as_prob_vector(p)


@validate_call(config=model_cfg, validate_return=True)
def state_smooth_probs(
    n: int,
    half_life: float = 5.0,
    kernel_type: int = 1,
    reference: float | None = None,
    time_based: bool = True,
) -> ProbVector:
    w = kernel_smoothing(
        n=n,
        half_life=half_life,
        kernel_type=kernel_type,
        reference=reference,
        time_based=time_based,
    )
    return as_prob_vector(w / w.sum())
