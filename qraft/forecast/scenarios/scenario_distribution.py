from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from polars import DataFrame

from scenarios.panel import ScenarioPanel
from scenarios.types import ProbVector, View


@dataclass(frozen=True)
class ScenarioDistribution:
    """Discrete scenario distribution, backed by an :class:`AssetPanel`.

    The panel owns all date-stripping and prob-length validation, so the
    public ``scenarios`` / ``prob`` / ``dates`` properties are thin views.
    """

    panel: ScenarioPanel

    @property
    def scenarios(self) -> DataFrame:
        """Asset-value matrix (no 'date' column)."""
        return self.panel.values

    @property
    def prob(self) -> ProbVector:
        return self.panel.prob

    @property
    def dates(self) -> pl.Series | None:
        """Row-aligned ``date`` series, or ``None`` if the panel has no dates."""
        return self.panel.dates

    @classmethod
    def default_instance(
        cls, scenarios: DataFrame, prob: ProbVector | None = None
    ) -> ScenarioDistribution:
        """Build from a raw DataFrame; uniform prior when *prob* is None."""
        return cls(panel=ScenarioPanel.from_frame(scenarios, prob))


@dataclass
class ScenarioProb:
    """Orchestrator for a :class:`ScenarioDistribution` and attached views.

    Currently holds both the original and current distributions plus a view
    list; entropy-pooling and CMA updates are delegated to their respective
    free functions / :class:`CopulaMarginalModel`.
    """

    _base_dist: ScenarioDistribution
    _dist: ScenarioDistribution
    views: list[View] = field(default_factory=list)

    @classmethod
    def default_inst(
        cls, scenarios: DataFrame, prob: ProbVector | None = None
    ) -> ScenarioProb:
        """Create a ScenarioProb with uniform prior when ``prob`` is None.

        The base and current distributions start pointing to the same
        underlying object.
        """
        dist = ScenarioDistribution.default_instance(scenarios=scenarios, prob=prob)
        return cls(_dist=dist, _base_dist=dist)

    @property
    def assets(self) -> list[str]:
        return self.scenarios.columns

    @property
    def scenarios(self) -> DataFrame:
        return self._dist.scenarios

    @property
    def prob(self) -> ProbVector:
        return self._dist.prob

    @property
    def dates(self) -> pl.Series | None:
        return self._dist.dates

    def add_views(self, new_views: list[View]) -> ScenarioProb:
        """Append views in place and return ``self`` for chaining."""
        self.views.extend(new_views)
        return self
