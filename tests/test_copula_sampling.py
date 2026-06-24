import numpy as np
import polars as pl
import pytest

from qraft.core.configs import CMAConfig
from qraft.core.probability import sampling


def test_cma_config_defaults_to_itau_copula_fit() -> None:
    cfg = CMAConfig(target_copula="t")

    assert cfg.copula_fit_method == "itau"


def test_sample_copula_does_not_fallback_to_ml_after_failed_method(
    monkeypatch,
) -> None:
    methods: list[str] = []

    class FailingStudentCopula:
        def __init__(self, dim: int) -> None:
            self.dim = dim

        def fit(self, values, method: str, to_pobs: bool):
            methods.append(method)
            raise TypeError("copulae analytical t fit failed")

    monkeypatch.setattr(sampling, "StudentCopula", FailingStudentCopula)

    with pytest.raises(RuntimeError, match="method 'itau'"):
        sampling.sample_copula(
            pl.DataFrame(
                {
                    "x": np.linspace(0.1, 0.9, 5),
                    "y": np.linspace(0.9, 0.1, 5),
                }
            ),
            parametric_copula="t",
            fit_method="itau",
        )

    assert methods == ["itau"]
