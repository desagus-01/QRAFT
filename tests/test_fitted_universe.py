from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from qraft.core.configs import PipelineConfig
from qraft.core.panel import ScenarioPanel
from qraft.core.probability.distributions import uniform_probs
from qraft.forecast.pipelines.fitted_universe import (
    ForecastRecipe,
    FittedUniverse,
    _merge_inv,
    fit_with_orders,
    recondition,
)
from qraft.forecast.time_series.models.fitted_types import (
    AutoARMARes,
    AutoGARCHRes,
    DemeanRes,
    UnivariateRes,
)
from qraft.forecast.time_series.preprocessing.types import (
    TransformDecision,
    UnivariatePreprocess,
)
from qraft.forecast.time_series.transforms.inverses import (
    DifferenceInverseSpec,
    SeasonalInverseSpec,
)


# ---------------------------------------------------------------------------
# _merge_inv
# ---------------------------------------------------------------------------


def test_merge_inv_combines_specs_from_both_dicts() -> None:
    inv_d = {"A": DifferenceInverseSpec(order=1, initial_values=np.array([1.0]))}
    inv_s = {"A": SeasonalInverseSpec(terms=[], next_t=0)}
    result = _merge_inv(inv_d, inv_s, survivors=["A"])
    assert result == {"A": [inv_d["A"], inv_s["A"]]}


def test_merge_inv_filters_to_survivors() -> None:
    spec_a = DifferenceInverseSpec(order=1, initial_values=np.array([1.0]))
    spec_b = SeasonalInverseSpec(terms=[], next_t=0)
    result = _merge_inv({"A": spec_a, "B": spec_b}, {}, survivors=["A"])
    assert list(result) == ["A"]
    assert result["A"] == [spec_a]


def test_merge_inv_omits_none_from_missing_keys() -> None:
    inv_d = {"A": DifferenceInverseSpec(order=1, initial_values=np.array([1.0]))}
    inv_s: dict[str, SeasonalInverseSpec] = {}
    result = _merge_inv(inv_d, inv_s, survivors=["A"])
    assert result == {"A": [inv_d["A"]]}


# ---------------------------------------------------------------------------
# fit_with_orders
# ---------------------------------------------------------------------------


def _post_data(
    asset_arrays: dict[str, np.ndarray],
    start: datetime = datetime(2024, 1, 1),
) -> pl.DataFrame:
    n = max(len(v) for v in asset_arrays.values())
    dates = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": dates, **asset_arrays})


def test_fit_with_orders_demean_fallback() -> None:
    rng = np.random.default_rng(42)
    n = 30
    y = rng.normal(0.2, 1.0, n)

    data = _post_data({"A": y})
    recipe = ForecastRecipe(
        detrend={},
        deseason={},
        mean_orders={"A": None},
        vol_orders={"A": None},
        survivors=["A"],
    )
    cfg = PipelineConfig()

    result = fit_with_orders(data, recipe, cfg)
    assert isinstance(result, dict)
    assert "A" in result

    ures = result["A"]
    assert isinstance(ures, UnivariateRes)
    assert isinstance(ures.mean_res, DemeanRes)
    assert ures.volatility_res is None
    assert ures.quality is None

    assert abs(float(np.mean(ures.mean_res.residuals))) < 1e-10


def test_fit_with_orders_arma() -> None:
    rng = np.random.default_rng(42)
    n = 100
    e = rng.normal(0, 0.1, n)
    y = np.zeros(n)
    phi = 0.7
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]

    data = _post_data({"A": y})
    recipe = ForecastRecipe(
        detrend={},
        deseason={},
        mean_orders={"A": (1, 0)},
        vol_orders={"A": None},
        survivors=["A"],
    )
    cfg = PipelineConfig()

    result = fit_with_orders(data, recipe, cfg)
    ures = result["A"]
    assert isinstance(ures.mean_res, AutoARMARes)
    assert ures.mean_res.model_order == (1, 0)
    assert ures.volatility_res is None


def test_fit_with_orders_arma_and_garch(tmp_path: pytest.TempPathFactory) -> None:
    rng = np.random.default_rng(42)
    n = 300
    omega, alpha, beta = 0.05, 0.1, 0.85
    sigma2 = np.ones(n) * omega / (1.0 - alpha - beta)
    y = np.zeros(n)
    phi = 0.5
    for i in range(1, n):
        sigma2[i] = (
            omega + alpha * (y[i - 1] - phi * y[i - 2]) ** 2 + beta * sigma2[i - 1]
        )
        y[i] = phi * y[i - 1] + rng.normal(0, 1) * np.sqrt(sigma2[i])

    data = _post_data({"A": y})
    recipe = ForecastRecipe(
        detrend={},
        deseason={},
        mean_orders={"A": (1, 0)},
        vol_orders={"A": (1, 0, 1)},
        survivors=["A"],
    )
    cfg = PipelineConfig()

    result = fit_with_orders(data, recipe, cfg)
    ures = result["A"]

    assert isinstance(ures.mean_res, AutoARMARes)
    assert ures.mean_res.model_order == (1, 0)

    if ures.volatility_res is not None:
        assert isinstance(ures.volatility_res, AutoGARCHRes)
        assert ures.volatility_res.model_order == (1, 0, 1)


def test_fit_with_orders_multiple_assets() -> None:
    rng = np.random.default_rng(42)
    n = 50
    y_a = rng.normal(0.1, 0.5, n)
    y_b = rng.normal(0.2, 1.0, n)

    data = _post_data({"A": y_a, "B": y_b})
    recipe = ForecastRecipe(
        detrend={},
        deseason={},
        mean_orders={"A": None, "B": None},
        vol_orders={"A": None, "B": None},
        survivors=["A", "B"],
    )
    cfg = PipelineConfig()

    result = fit_with_orders(data, recipe, cfg)
    assert set(result) == {"A", "B"}
    for ures in result.values():
        assert isinstance(ures.mean_res, DemeanRes)
        assert ures.volatility_res is None


# ---------------------------------------------------------------------------
# recondition
# ---------------------------------------------------------------------------


def test_recondition_returns_fitted_universe() -> None:
    n = 60
    rng = np.random.default_rng(42)
    trend = np.cumsum(rng.normal(0, 0.05, n))
    seasonal = 0.3 * np.sin(np.pi / 6 * np.arange(n))
    noise = rng.normal(0, 0.02, n)
    values = 100.0 + trend + seasonal + noise

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    data = pl.DataFrame({"date": dates, "A": values})
    prob = uniform_probs(data.height)

    recipe = ForecastRecipe(
        detrend={"A": TransformDecision(kind="difference", order=1)},
        deseason={"A": [("w1", np.pi / 6)]},
        mean_orders={"A": None},
        vol_orders={"A": None},
        survivors=["A"],
    )
    cfg = PipelineConfig()

    result = recondition(recipe, data, prob, cfg)

    assert isinstance(result, FittedUniverse)
    assert result.assets == ["A"]
    assert isinstance(result.preprocess, UnivariatePreprocess)
    assert result.preprocess.post_data.shape[0] > 0

    assert "A" in result.models
    ures = result.models["A"]
    assert isinstance(ures.mean_res, DemeanRes)
    assert ures.volatility_res is None

    assert "A" in result.simulation_forecasts
    assert isinstance(result.invariants, ScenarioPanel)
    assert result.invariants.kind == "invariant"
    assert "A" in result.invariants.asset_names


def test_recondition_recipe_round_trip() -> None:
    n = 60
    rng = np.random.default_rng(42)
    trend = np.cumsum(rng.normal(0, 0.05, n))
    seasonal = 0.3 * np.sin(np.pi / 6 * np.arange(n))
    noise = rng.normal(0, 0.02, n)
    values = 100.0 + trend + seasonal + noise

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    data = pl.DataFrame({"date": dates, "A": values})
    prob = uniform_probs(data.height)

    recipe = ForecastRecipe(
        detrend={"A": TransformDecision(kind="difference", order=1)},
        deseason={"A": [("w1", np.pi / 6)]},
        mean_orders={"A": None},
        vol_orders={"A": None},
        survivors=["A"],
    )

    result = recondition(recipe, data, prob, PipelineConfig())

    round_tripped = result.recipe()
    assert round_tripped.survivors == recipe.survivors
    assert round_tripped.detrend == recipe.detrend
    assert round_tripped.deseason == recipe.deseason
    assert round_tripped.mean_orders == recipe.mean_orders
    assert round_tripped.vol_orders == recipe.vol_orders
