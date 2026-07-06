from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from qraft.core.probability.prob_vector import ProbVector

Sign: TypeAlias = Literal["<=", ">=", "=="]


@dataclass(frozen=True)
class MeanView:
    asset: str
    sign: Sign
    target: float


@dataclass(frozen=True)
class StdView:
    asset: str
    sign: Sign
    target: float


@dataclass(frozen=True)
class CorrView:
    pair: tuple[str, str]
    sign: Sign
    target: float


@dataclass(frozen=True)
class RankingView:
    """Rank assets by view-panel values.

    View panels use simple per-period returns; rankings can differ from log-return
    rankings for large divergent moves.
    """

    order: list[str]

    def __post_init__(self) -> None:
        if len(self.order) < 2:
            raise ValueError(
                "RankingView needs at least two assets to express an ordering; got 1."
            )
        if len(set(self.order)) != len(self.order):
            raise ValueError("RankingView cannot contain duplicate assets.")


@dataclass(frozen=True)
class QuantileView:
    asset: str
    quantile: float
    target_prob: float


ViewSpec: TypeAlias = MeanView | StdView | CorrView | RankingView | QuantileView


class ConstraintDiag(TypedDict):
    risk_driver: tuple[str, str] | str
    sign: Sign  # was ConstraintSignLike
    constraint_value: float | None  # was NDArray | None
    active: bool
    sensitivity: float | None


@dataclass(frozen=True, slots=True)
class ViewDiagnostics:
    ens_prior: float
    ens_posterior: float
    constraints: tuple[ConstraintDiag, ...]
    solver_status: str
    ens_collapsed: bool


@dataclass(frozen=True, slots=True)
class EntropyPoolingResult:
    posterior: ProbVector
    diagnostics: ViewDiagnostics
