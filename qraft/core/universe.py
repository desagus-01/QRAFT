from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AssetSubset = Literal["all", "tradable", "factors"]


@dataclass(frozen=True, slots=True)
class AssetUniverse:
    """Classifies tickers as tradable assets or non-tradable factors.

    Both assets and factors are forecast together (preserving cross-correlations),
    but only assets participate in portfolio construction (weights, PnL, value).
    """

    assets: list[str]
    factors: list[str]

    def __post_init__(self) -> None:
        overlap = set(self.assets) & set(self.factors)
        if overlap:
            raise ValueError(
                f"Tickers appear in both assets and factors: {sorted(overlap)}"
            )
        if not self.assets:
            raise ValueError("Must have at least one tradable asset")

    @classmethod
    def factors_free(cls, assets: list[str]) -> "AssetUniverse":
        return cls(assets=assets, factors=[])

    @property
    def all_tickers(self) -> list[str]:
        """All tickers in forecast order (assets first, then factors)."""
        return self.assets + self.factors

    def is_factor(self, ticker: str) -> bool:
        return ticker in set(self.factors)

    def is_asset(self, ticker: str) -> bool:
        return ticker in set(self.assets)
