import numpy as np
import polars as pl
import pytest

from qraft.forecast.signals.raw_signals import Signal, ewma_signal
from qraft.forecast.signals.signal_processing import (
    process_signal,
    signal_distortion,
    signal_ranking,
    signal_scoring,
    signal_smoothing,
)


def test_ewma_signal_differences_before_smoothing_and_can_preserve_nulls() -> None:
    dates = pl.Series("date", ["2024-01-01", "2024-01-02", "2024-01-03"]).str.to_date()
    risk_drivers = pl.DataFrame({"A": [1.0, 3.0, 6.0]})

    signal = ewma_signal(risk_drivers, dates, half_life="1d", drop_nulls=False)

    assert signal.type == "momentum"
    assert signal.values.height == 3
    assert signal.values["A"][0] is None
    assert signal.values["A"][1] == 2.0
    assert signal.values["A"][2] > 2.0

    assert (
        ewma_signal(risk_drivers, dates, half_life="1d", drop_nulls=True).values.height
        == 2
    )


def test_signal_smoothing_preserves_shape_and_columns() -> None:
    df = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]})

    smoothed = signal_smoothing(df, half_life=2)

    assert smoothed.shape == df.shape
    assert smoothed.columns == df.columns


def test_signal_scoring_drops_initial_nan_and_returns_finite_values() -> None:
    df = pl.DataFrame({"A": [1.0, 2.0, 4.0, 8.0], "B": [8.0, 4.0, 2.0, 1.0]})

    scored = signal_scoring(df, half_life=2)

    assert scored.height < df.height
    assert np.all(np.isfinite(scored.to_numpy()))


def test_signal_ranking_is_row_wise_cross_sectional() -> None:
    df = pl.DataFrame({"A": [1.0, 3.0], "B": [2.0, 2.0], "C": [3.0, 1.0]})

    ranked = signal_ranking(df)

    assert ranked.row(0) == (-1.0, 0.0, 1.0)
    assert ranked.row(1) == (1.0, 0.0, -1.0)


def test_signal_ranking_requires_at_least_two_columns() -> None:
    with pytest.raises(ValueError, match="at least two"):
        signal_ranking(pl.DataFrame({"A": [1.0, 2.0]}))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("identity", [-0.5, 0.0, 0.5]),
        ("terciles", [-1.0, 0.0, 1.0]),
        ("median", [-1.0, 0.0, 1.0]),
        ("winsorize", [-0.4, 0.0, 0.4]),
        ("power", [-0.25, 0.0, 0.25]),
    ],
)
def test_signal_distortion_methods(kind, expected) -> None:
    out = signal_distortion(
        pl.DataFrame({"A": [-0.5], "B": [0.0], "C": [0.5]}),
        kind,
        winsor_limit=0.4,
        power=2.0,
    )

    assert out.row(0) == pytest.approx(tuple(expected))


def test_signal_distortion_smooth_methods_preserve_sign_and_bounds() -> None:
    df = pl.DataFrame({"A": [-1.0], "B": [0.0], "C": [1.0]})

    for kind in ("tanh", "arctan"):
        row = signal_distortion(df, kind, strength=3.0).row(0)
        assert row[0] == pytest.approx(-1.0)
        assert row[1] == pytest.approx(0.0)
        assert row[2] == pytest.approx(1.0)


def test_signal_distortion_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown distortion"):
        signal_distortion(pl.DataFrame({"A": [1.0], "B": [-1.0]}), "bad")  # type: ignore[arg-type]


def test_process_signal_skips_smoothing_for_momentum() -> None:
    values = pl.DataFrame({"A": [1.0, 2.0, 4.0, 8.0], "B": [8.0, 4.0, 2.0, 1.0]})

    direct = signal_distortion(signal_ranking(signal_scoring(values, 2)), "identity")
    processed = process_signal(Signal("momentum", values), 99, 2, "average", "identity")

    assert processed.equals(direct)


def test_process_signal_smooths_non_momentum_signal() -> None:
    values = pl.DataFrame(
        {
            "A": [1.0, 4.0, 2.0, 8.0, 3.0],
            "B": [3.0, 1.0, 5.0, 2.0, 6.0],
            "C": [2.0, 3.0, 1.0, 7.0, 4.0],
        }
    )

    unsmoothed = signal_distortion(
        signal_ranking(signal_scoring(values, 2)), "identity"
    )
    processed = process_signal(Signal("value", values), 2, 2, "average", "identity")

    assert not processed.equals(unsmoothed)
