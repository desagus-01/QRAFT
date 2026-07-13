from qraft.core.configs import MeanModelConfig
from qraft.forecast.time_series.models.mean import _arma_filter


def test_arma_filter_rejects_roots_inside_stationarity_buffer() -> None:
    cfg = MeanModelConfig(arma_stationarity_buffer=1e-3)

    assert not _arma_filter({"ar.L1": 1.0}, cfg)
    assert not _arma_filter({"ar.L1": 1.0005}, cfg)


def test_arma_filter_rejects_roots_inside_invertibility_buffer() -> None:
    cfg = MeanModelConfig(arma_invertibility_buffer=1e-3)

    assert not _arma_filter({"ma.L1": -1.0}, cfg)
    assert not _arma_filter({"ma.L1": -1.0005}, cfg)
