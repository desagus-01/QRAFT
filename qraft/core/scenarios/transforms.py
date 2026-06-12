from dataclasses import dataclass
from typing import Protocol, Sequence

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.entropy_pooling import entropy_pooling_probs
from qraft.core.scenarios.copula_marginal import CMAConfig, CopulaMarginalModel
from qraft.core.scenarios.view_types import ViewSpec


class ScenarioTransform(Protocol):
    def apply(self, panel: ScenarioPanel) -> ScenarioPanel: ...


@dataclass(frozen=True)
class Views:
    specs: list[ViewSpec]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0 and 1, currently {self.confidence}"
            )

    def apply(self, panel: ScenarioPanel) -> ScenarioPanel:
        posterior = entropy_pooling_probs(
            panel=panel, specs=self.specs, confidence=self.confidence
        )
        return panel.with_prob(posterior)


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
