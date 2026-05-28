from dataclasses import dataclass
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy._typing._array_like import NDArray
from scipy.stats import f as f_dist
from typing_extensions import Literal

from time_series.tests.types import HypTestRes, format_hyp_test_result

SEASONAL_PERIODS = Literal["weekly", "monthly", "quarterly", "semi-annual", "annual"]

SEASONAL_MAP = {
    "weekly": 5,
    "monthly": 21,
    "semi-annual": 126,
    "annual": 252,
    "quarterly": 63,
}


class SeasonalityPeriodTest(NamedTuple):
    seasonal_period: str
    seasonal_frequency_radian: float
    evidence_of_seasonality: bool
    res: HypTestRes


class SeasonalBin(NamedTuple):
    harmonic: int
    power: float
    idx: int
    frequency: float
    period: float


@dataclass(frozen=True)
class Periodogram:
    power: NDArray[np.floating]
    freq_cycles: NDArray[np.floating]
    sample_count: int


class FStatRes(NamedTuple):
    stat: float
    numerator_degree_of_freedom: int
    denominator_degree_of_freedom: int


def plot_periodogram(
    data: NDArray[np.floating],
    max_period: int = 260,
    seasonal_map: dict[str, int] = SEASONAL_MAP,
    show_labels: bool = True,
) -> Axes:
    periodo = periodogram(data=data, need_demean=False)
    mask = periodo.freq_cycles > 0
    periods = 1.0 / periodo.freq_cycles[mask]
    spec = periodo.power[mask]

    fig, ax = plt.subplots()
    ax.plot(periods, spec)

    # Add seasonal reference lines
    for name, td in seasonal_map.items():
        if 1 <= td <= max_period:
            ax.axvline(td, linestyle="--", linewidth=1, alpha=0.8)
            if show_labels:
                ax.text(
                    td,
                    0.98,
                    name,
                    rotation=90,
                    va="top",
                    ha="right",
                    transform=ax.get_xaxis_transform(),
                    fontsize=9,
                    alpha=0.9,
                )

    ax.set_xlim(1, max_period)
    ax.set_xlabel("Trading Days per cycle")
    ax.set_ylabel("Power")
    ax.set_title("Periodogram")
    return ax


def _make_len_multiple_of_seasonal_period(
    data: NDArray[np.floating], seasonal_period: int
) -> NDArray[np.floating]:
    n = data.shape[0]
    extra = n % seasonal_period
    if extra == 0:
        return data
    return data[extra:]


def periodogram(data: NDArray[np.floating], need_demean: bool = False) -> Periodogram:
    if need_demean:
        data = data - data.mean()
    sample_count = data.size

    total_sum = data.sum()
    total_energy = float(np.dot(data, data))

    if total_energy == 0.0:
        power = np.zeros(1 + sample_count // 2, dtype=float)
    else:
        spectrum = np.fft.rfft(data)
        power = np.empty(spectrum.shape, dtype=float)

        # DC component
        power[0] = (total_sum * total_sum) / total_energy

        # Interior frequencies
        if sample_count % 2 == 0:  # even length: has Nyquist bin
            if power.size > 2:
                power[1:-1] = 2.0 * (np.abs(spectrum[1:-1]) ** 2) / total_energy
            # Nyquist frequency (no doubling)
            power[-1] = (np.abs(spectrum[-1]) ** 2) / total_energy
        else:  # odd length: no Nyquist bin
            if power.size > 1:
                power[1:] = 2.0 * (np.abs(spectrum[1:]) ** 2) / total_energy

    # Frequency grids
    k = np.arange(power.size, dtype=float)
    freq_cycles = k / sample_count

    return Periodogram(
        power=power,
        freq_cycles=freq_cycles,
        sample_count=sample_count,
    )


def get_seasonal_bins(
    power: NDArray[np.floating],
    frequency: NDArray[np.floating],
    n_samples: int,
    seasonal_period: int,
) -> list[SeasonalBin]:
    if n_samples % seasonal_period != 0:
        raise ValueError(
            f"n_samples={n_samples} must be a multiple of cycle_period={seasonal_period} "
        )

    n_harmonics = (seasonal_period - 1) // 2
    cycles_in_sample = n_samples // seasonal_period

    seasonal_bins: list[SeasonalBin] = []
    for k in range(1, n_harmonics + 1):
        idx = k * cycles_in_sample
        pwr = float(power[idx])
        f = float(frequency[idx])
        seasonal_bins.append(
            SeasonalBin(harmonic=k, power=pwr, idx=idx, frequency=f, period=1.0 / f)
        )

    return seasonal_bins


def get_period_fstat(periodogram: Periodogram, seasonal_period: int) -> FStatRes:
    seasonal_bins = get_seasonal_bins(
        power=periodogram.power,
        frequency=periodogram.freq_cycles,
        n_samples=periodogram.sample_count,
        seasonal_period=seasonal_period,
    )
    seasonal_period_power = np.asarray([bin.power for bin in seasonal_bins]).sum()

    n_harmonics = (seasonal_period - 1) // 2
    degrees_of_freedom = 2 * n_harmonics

    # Needs Nyquist harmonic if seasonal period is even
    if seasonal_period % 2 == 0:
        degrees_of_freedom += 1
        seasonal_period_power += periodogram.power[-1]

    remaining_degrees_of_freedom = (
        periodogram.sample_count - degrees_of_freedom - 1
    )  # 1 here accounts for intercept

    f_stat_numerator = remaining_degrees_of_freedom * seasonal_period_power
    f_stat_denominator = (
        periodogram.sample_count - seasonal_period_power - periodogram.power[0]
    ) * degrees_of_freedom

    return FStatRes(
        stat=f_stat_numerator / f_stat_denominator,
        numerator_degree_of_freedom=degrees_of_freedom,
        denominator_degree_of_freedom=remaining_degrees_of_freedom,
    )


def get_periodogram_p_val(
    periodogram: Periodogram, seasonal_period: int
) -> tuple[float, float]:
    t_stat = get_period_fstat(periodogram=periodogram, seasonal_period=seasonal_period)
    return t_stat.stat, f_dist.sf(
        t_stat.stat,
        t_stat.numerator_degree_of_freedom,
        t_stat.denominator_degree_of_freedom,
    )


def periodogram_seasonality_test(
    data: NDArray[np.floating], seasonal_period: SEASONAL_PERIODS
) -> SeasonalityPeriodTest:
    seasonal_period_n = SEASONAL_MAP[seasonal_period]
    data = _make_len_multiple_of_seasonal_period(
        data=data, seasonal_period=seasonal_period_n
    )
    period = periodogram(data=data, need_demean=False)
    stat, p_val = get_periodogram_p_val(
        periodogram=period, seasonal_period=seasonal_period_n
    )

    hypothesis_test_res = format_hyp_test_result(
        p_val=p_val, stat=stat, null=f"No {seasonal_period} seasonality"
    )

    return SeasonalityPeriodTest(
        seasonal_period=seasonal_period,
        seasonal_frequency_radian=2 * np.pi / SEASONAL_MAP[seasonal_period],
        evidence_of_seasonality=hypothesis_test_res.reject_null,
        res=hypothesis_test_res,
    )
