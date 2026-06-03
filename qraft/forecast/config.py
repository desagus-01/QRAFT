from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

GarchDist = Literal["t", "normal"]
IC = Literal["bic", "aic"]


@dataclass(frozen=True, slots=True)
class MeanModelConfig:
    """Thresholds and search limits for ARMA mean modelling."""

    # Candidate search space
    max_ar_order: int = 2
    max_ma_order: int = 2
    search_n_models: int = 5
    information_criteria: IC = "bic"

    # Root-distance buffers from the unit circle (admissibility)
    arma_stationarity_buffer: float = 1e-3
    arma_invertibility_buffer: float = 1e-3

    # Ljung-Box lags tested when screening for residual autocorrelation.
    # Tuple enforces immutability on the frozen dataclass.
    ljung_box_lags: tuple[int, ...] = (10, 15, 20)

    # Minimum number of rejected lags to flag a series as needing a model
    min_ljung_box_rejections: int = 1


@dataclass(frozen=True, slots=True)
class VolatilityModelConfig:
    """Thresholds and search limits for GARCH volatility modelling."""

    # Candidate search space
    max_p_order: int = 2
    max_o_order: int = 1
    max_q_order: int = 2
    candidate_distributions: tuple[GarchDist, ...] = ("t", "normal")

    # Admissibility constraints
    max_persistence: float = 0.995
    tolerance_zero: float = 1e-10
    tolerance_dups: float = 1e-6

    # Residual diagnostic lags / rejection thresholds.
    # Tuples enforce immutability on the frozen dataclass.
    ljung_box_lags: tuple[int, ...] = (10, 20)
    arch_lags: tuple[int, ...] = (5, 10, 15)
    min_ljung_box_rejections: int = 2
    min_arch_rejections: int = 1


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Penalty weights and grade cut-offs for SelectionAudit scoring."""

    penalty_high: float = 30.0
    penalty_medium: float = 20.0
    penalty_low: float = 10.0

    # Score thresholds (inclusive lower bound) for grade assignment
    grade_a_threshold: float = 85.0
    grade_b_threshold: float = 70.0
    grade_c_threshold: float = 50.0


@dataclass(frozen=True, slots=True)
class IIDConfig:
    """Lags and significance level for white-noise IID screens."""

    lags_simple: int = 10
    lags_complex: int = 5
    significance_level: float = 0.05
    mc_iters: int = 2000
    perm_test_iters: int = 200


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Controls for the univariate preprocessing pipeline."""

    trend_order_max: int = 3
    trend_threshold_order: int = 2
    trend_type: Literal["deterministic", "stochastic", "both"] = "both"
    iid: IIDConfig = field(default_factory=IIDConfig)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Single configuration object for top-level pipeline entry points.

    Pass an instance to ``get_univariate_results`` or
    ``run_univariate_preprocess``.  Override any sub-config; the rest keep
    their defaults::

        cfg = PipelineConfig(
            mean=MeanModelConfig(max_ar_order=3),
            quality=QualityConfig(penalty_high=40.0),
        )
    """

    mean: MeanModelConfig = field(default_factory=MeanModelConfig)
    volatility: VolatilityModelConfig = field(default_factory=VolatilityModelConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


DEFAULT_PIPELINE_CONFIG: PipelineConfig = PipelineConfig()


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Controls centralised logging for the funcasting package.

    Attributes
    ----------
    level :
        Log level for the funcasting logger (default: INFO).
    log_file :
        Optional path to write logs to. If None, logs go to stderr only.
    third_party_level :
        Level applied to noisy third-party loggers (arch, statsmodels,
        matplotlib). Default WARNING keeps fitting noise out of the way.
    fmt :
        Log format string. The default omits timestamps for interactive use;
        timestamps are added automatically when ``log_file`` is set.
    """

    level: int = logging.INFO
    log_file: str | None = None
    third_party_level: int = logging.WARNING
    fmt: str = "[%(levelname)s] %(name)s - %(message)s"


DEFAULT_LOG_CONFIG: LogConfig = LogConfig()
