from dataclasses import dataclass

import numpy as np
from construction.optimization.presets import PreMadeObjectives
from forecast.scenarios.types import ProbVector
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PortflioExecution:
    forecast_values: NDArray[np.floating]
    rebalance_frequency: int
    path_probs: ProbVector
    policy: PreMadeObjectives
