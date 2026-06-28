from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qraft.backtest.schedule import RebalanceSchedule

GarchDist = Literal["t", "normal", "skewt"]
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
    ljung_box_lags: tuple[int, ...] = (1, 2, 5, 10, 20)
    ljung_box_test_is_robust: bool = True

    # Minimum number of rejected lags to flag a series as needing a model
    min_ljung_box_rejections: int = 1


@dataclass(frozen=True, slots=True)
class VolatilityModelConfig:
    """Thresholds and search limits for GARCH volatility modelling."""

    # Candidate search space
    max_p_order: int = 2
    min_o_order: int = 0
    max_o_order: int = 1
    max_q_order: int = 2
    candidate_distributions: tuple[GarchDist, ...] = ("t", "skewt")

    # Admissibility constraints
    max_persistence: float = 0.9999
    variance_cap_factor: float = 4.0
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
    lags_complex: int = 3
    significance_level: float = 0.05
    mc_iters: int = 1000
    perm_test_iters: int = 1000
    perm_test_min_iters: int = 100
    perm_test_batch_iters: int = 50
    perm_test_ci_level: float = 0.99
    ellipsoid_rho_threshold: float = 0.10
    arch_lags: tuple[int, ...] = (10,)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Controls for the univariate preprocessing pipeline."""

    trend_order_max: int = 3
    trend_threshold_order: int = 2
    trend_type: Literal["deterministic", "stochastic", "both"] = "both"
    iid: IIDConfig = field(default_factory=IIDConfig)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Single configuration object for top-level pipeline entry points."""

    mean: MeanModelConfig = field(default_factory=MeanModelConfig)
    volatility: VolatilityModelConfig = field(default_factory=VolatilityModelConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    exclude_non_invariants: bool = True


DEFAULT_PIPELINE_CONFIG: PipelineConfig = PipelineConfig()


SamplingMethod = Literal["bootstrap", "historical", "cma"]
SelectionMetric = Literal[
    "sharpe",
    "sortino",
    "calmar",
    "total_return",
    "annualised_return",
    "cvar",
]


@dataclass(frozen=True)
class CMAConfig:
    target_copula: Literal["t", "norm"] | None = None
    target_marginals: dict[str, Literal["t", "norm"]] | None = None
    copula_fit_method: Literal["ml", "irho", "itau"] = "itau"

    def __post_init__(self) -> None:
        if self.target_copula is None and self.target_marginals is None:
            raise ValueError(
                "CMAConfig needs at least one of target_copula or target_marginals."
            )


@dataclass(frozen=True, slots=True)
class SimulationForecastConfig:
    horizon: int = 10
    n_sims: int = 1000
    method: SamplingMethod = "bootstrap"
    back_to_price: bool = True
    cma_config: CMAConfig | None = None


DEFAULT_SIMULATION_CONFIG: SimulationForecastConfig = SimulationForecastConfig()


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Controls for walk-forward selection experiments."""

    train_size: int | None = None
    test_size: int | None = None
    step: int | None = None
    embargo: int = 0
    anchored: bool = False
    schedule: RebalanceSchedule = field(default_factory=RebalanceSchedule)
    step_size: int = 1
    initial_cash: float = 100.0
    periods_per_year: float = 252.0
    risk_free_rate: float = 0.0
    metric: SelectionMetric = "sharpe"
    max_held_fraction: float = 0.5
    pbo_blocks: int = 10

    def __post_init__(self) -> None:
        if self.train_size is None or self.test_size is None:
            raise ValueError("train_size and test_size are required")
        if self.train_size < 1 or self.test_size < 1:
            raise ValueError("train_size and test_size must be >= 1")
        if self.step is not None and self.step < 1:
            raise ValueError("step must be >= 1")
        if self.embargo < 0:
            raise ValueError("embargo must be >= 0")
        if self.step_size < 1:
            raise ValueError("step_size must be >= 1")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")
        if not 0 <= self.max_held_fraction <= 1:
            raise ValueError("max_held_fraction must be between 0 and 1")
        if self.pbo_blocks < 2:
            raise ValueError("pbo_blocks must be >= 2")


DEFAULT_WALK_FORWARD_CONFIG: WalkForwardConfig = WalkForwardConfig(
    train_size=1,
    test_size=1,
)
