from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.entropy_pooling import entropy_pooling_probs
from qraft.core.scenarios.copula_marginal import CMAConfig, CopulaMarginalModel
from qraft.core.scenarios.view_types import ViewSpec


class ScenarioTransform(Protocol):
    def apply(self, panel: ScenarioPanel) -> ScenarioPanel: ...


@dataclass(frozen=True)
class Views:
    """Entropy-pooling view transform.

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
        result = entropy_pooling_probs(
            panel=panel,
            specs=self.specs,
            confidence=self.confidence,
            solver=self.solver,
            **(self.solver_kwargs or {}),
        )
        return panel.with_prob(result.posterior)


@dataclass(frozen=True)
class CMA:
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
