from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

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
    order: list[str]


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
