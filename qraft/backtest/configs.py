from __future__ import annotations

from dataclasses import dataclass, field

from qraft.core.configs import SelectionMetric
from qraft.core.schedule import RebalanceSchedule


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Execution and scoring settings shared by backtest entry points."""

    schedule: RebalanceSchedule = field(default_factory=RebalanceSchedule)
    initial_cash: float = 100.0
    periods_per_year: float | None = None

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        if self.periods_per_year is not None and self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")


@dataclass(frozen=True, slots=True)
class CVConfig:
    """Base configuration for cross-validation selection experiments."""

    risk_free_rate: float = 0.0
    metric: SelectionMetric = "sharpe"
    max_held_fraction: float = 0.5
    pbo_blocks: int = 10

    def __post_init__(self) -> None:
        if not 0 <= self.max_held_fraction <= 1:
            raise ValueError("max_held_fraction must be between 0 and 1")
        if self.pbo_blocks < 2:
            raise ValueError("pbo_blocks must be >= 2")


@dataclass(frozen=True, slots=True)
class WalkForwardConfig(CVConfig):
    """Controls for walk-forward selection experiments."""

    train_size: int = 12
    test_size: int = 2
    fold_step: int | None = None
    embargo: int = 0
    anchored: bool = False

    def __post_init__(self) -> None:
        CVConfig.__post_init__(self)
        if self.train_size is None or self.test_size is None:
            raise ValueError("train_size and test_size are required")
        if self.train_size < 1:
            raise ValueError("train_size must be >= 1")
        if self.test_size < 2:
            raise ValueError(
                "test_size must be >= 2 so each validation window can produce returns"
            )
        if self.fold_step is not None and self.fold_step < 1:
            raise ValueError("fold_step must be >= 1")
        if self.embargo < 0:
            raise ValueError("embargo must be >= 0")


@dataclass(frozen=True, slots=True)
class CombinatorialCVConfig:
    """Controls for combinatorial purged cross-validation (CPCV) selection."""

    n_groups: int = 10
    n_test_groups: int = 3
    purge: int = 1
    embargo: int = 1
    risk_free_rate: float = 0.0
    metric: SelectionMetric = "sharpe"
    max_held_fraction: float = 0.5
    pbo_blocks: int = 10

    def __post_init__(self) -> None:
        CVConfig.__post_init__(self)
        if self.n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if self.n_test_groups < 1 or self.n_test_groups >= self.n_groups:
            raise ValueError("n_test_groups must be between 1 and n_groups - 1")
        if self.purge < 0:
            raise ValueError("purge must be >= 0")
        if self.embargo < 0:
            raise ValueError("embargo must be >= 0")

    @property
    def cv_config(self) -> CVConfig:
        return CVConfig(
            risk_free_rate=self.risk_free_rate,
            metric=self.metric,
            max_held_fraction=self.max_held_fraction,
            pbo_blocks=self.pbo_blocks,
        )
