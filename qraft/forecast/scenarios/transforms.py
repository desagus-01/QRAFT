from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from forecast.scenarios.copula_marginal import CopulaMarginalModel
from forecast.scenarios.entropy_pooling import entropy_pooling_probs
from forecast.scenarios.panel import ScenarioPanel
from forecast.scenarios.types import ViewSpec


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
    target_copula: Literal["t", "norm"] | None = None
    target_marginals: dict[str, Literal["t", "norm"]] | None = None
    copula_fit_method: Literal["ml", "irho", "itau"] = "itau"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.target_copula is None and self.target_marginals is None:
            raise ValueError(
                "CMA needs at least one of target_copula or target_marginals"
            )

    def apply(self, panel: ScenarioPanel) -> ScenarioPanel:
        clean = panel.drop_nulls()
        return CopulaMarginalModel.from_panel(clean).update_distribution(
            seed=self.seed,
            target_marginals=self.target_marginals,
            target_copula=self.target_copula,
            copula_fit_method=self.copula_fit_method,
        )


def apply_scenario_transforms(
    panel: ScenarioPanel, transforms: Sequence[ScenarioTransform]
) -> ScenarioPanel:
    for transform in transforms:
        panel = transform.apply(panel)
    return panel
