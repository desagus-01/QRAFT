"""Scenario transforms for entropy views and copula-marginal adjustment."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Sequence

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.entropy_pooling import entropy_pooling
from qraft.core.scenarios.copula_marginal import CMAConfig, CopulaMarginalModel
from qraft.core.scenarios.view_types import ViewSpec
from qraft.core.scenarios.views import ViewedDistribution


class ScenarioTransform(Protocol):
    def apply(self, panel: ScenarioPanel) -> ScenarioPanel: ...


@dataclass(frozen=True)
class Views:
    """Entropy-pooling view transform over simple per-period returns.

    ``confidence`` is the linear posterior blend weight: ``1.0`` keeps the full
    EP posterior and ``0.0`` keeps the prior panel probabilities.
    """

    specs: list[ViewSpec]
    confidence: float = 1.0
    solver: str = "SCS"
    solver_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0 and 1, currently {self.confidence}"
            )

    def apply(self, panel: ScenarioPanel) -> ScenarioPanel:
        """Return ``panel`` with entropy-pooled posterior probabilities applied."""
        viewed = self.view_distribution(panel, as_of=panel.dates[-1])
        return viewed.panel.with_prob(viewed.posterior)

    def view_distribution(
        self, panel: ScenarioPanel, *, as_of: datetime
    ) -> ViewedDistribution:
        """Return prior, posterior, and diagnostics for applying views to ``panel``."""
        if panel.kind != "return":
            raise ValueError("Views must be applied to a simple-return panel")
        result = entropy_pooling(
            panel=panel,
            specs=self.specs,
            confidence=self.confidence,
            solver=self.solver,
            **(self.solver_kwargs or {}),
        )
        return ViewedDistribution(
            panel=panel,
            posterior=result.posterior,
            prior=panel.prob,
            diagnostics=result.diagnostics,
            as_of=as_of,
        )

    def against(self, market: Any) -> ViewedDistribution:
        """Return viewed returns by applying this transform against market data."""
        return market.viewed_returns(self)


@dataclass(frozen=True)
class CMA:
    """Copula-marginal adjustment transform for scenario panels."""

    config: CMAConfig
    seed: int | None = None

    def apply(self, panel: ScenarioPanel) -> ScenarioPanel:
        clean: ScenarioPanel = panel.drop_nulls()
        return CopulaMarginalModel.from_panel(clean).update_distribution(
            self.config, seed=self.seed, use_weighted_fit=True
        )


def apply_scenario_transforms(
    panel: ScenarioPanel, transforms: Sequence[ScenarioTransform]
) -> ScenarioPanel:
    for transform in transforms:
        panel = transform.apply(panel)
    return panel
