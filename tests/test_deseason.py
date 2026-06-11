import numpy as np
import polars as pl

from qraft.forecast.time_series.preprocessing.apply import apply_deseason
from qraft.forecast.time_series.transforms.deseason import (
    deterministic_seasonal_adjustment,
)
from qraft.forecast.time_series.transforms.inverses import SeasonalInverseSpec


def test_deterministic_seasonal_adjustment_keeps_asset_specific_time_origin() -> None:
    omega = 2.0 * np.pi / 5.0
    t = np.arange(10, dtype=float)
    data = pl.DataFrame(
        {
            "date": list(range(10)),
            "asset": np.cos(omega * t),
            "other": [None, *np.arange(1.0, 10.0).tolist()],
        }
    )

    res = deterministic_seasonal_adjustment(data, "asset", [omega])

    assert res.next_t == data.height
    assert res.residuals.height == data.height
    np.testing.assert_allclose(
        res.residuals.get_column("asset").to_numpy(),
        np.zeros(data.height),
        atol=1e-12,
    )


def test_apply_deseason_stores_future_phase_on_inverse_spec() -> None:
    omega = 2.0 * np.pi / 5.0
    t = np.arange(10, dtype=float)
    data = pl.DataFrame(
        {
            "date": list(range(10)),
            "asset": np.cos(omega * t),
        }
    )

    _, specs = apply_deseason(data, {"asset": [("weekly", omega)]})

    spec = specs["asset"]
    assert isinstance(spec, SeasonalInverseSpec)
    assert spec.next_t == data.height

    restored = spec.inverse_for_forecasts(np.zeros((1, 2)))
    expected = np.cos(omega * np.arange(data.height, data.height + 2))
    np.testing.assert_allclose(restored, expected.reshape(1, -1), atol=1e-12)
