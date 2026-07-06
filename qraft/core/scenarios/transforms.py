from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from qraft.core.panel import (
    ScenarioPanel,
    expand_posterior_to_parent,
)
from qraft.core.probability.entropy_pooling import entropy_pooling_probs
from qraft.core.probability.prob_vector import ProbVector
from qraft.core.scenarios.copula_marginal import CMAConfig, CopulaMarginalModel
from qraft.core.scenarios.view_types import ViewDiagnostics, ViewSpec
from qraft.utils.helpers import weighted_moments


class ScenarioTransform(Protocol):
    def apply(self, panel: ScenarioPanel) -> ScenarioPanel: ...


@dataclass(frozen=True)
class ViewedDistribution:
    panel: ScenarioPanel
    posterior: ProbVector
    prior: ProbVector
    diagnostics: ViewDiagnostics
    as_of: datetime

    def moments(self) -> dict[str, dict[str, float]]:
        values = self.panel.values.to_numpy().T
        prior_mean, prior_sd = weighted_moments(values, self.prior)
        posterior_mean, posterior_sd = weighted_moments(values, self.posterior)
        return {
            asset: {
                "prior_mean": float(prior_mean[i]),
                "posterior_mean": float(posterior_mean[i]),
                "prior_std": float(prior_sd[i]),
                "posterior_std": float(posterior_sd[i]),
            }
            for i, asset in enumerate(self.panel.asset_names)
        }

    def prob_for(self, parent: ScenarioPanel) -> ProbVector:
        return expand_posterior_to_parent(self.posterior, parent.prob, 1)

    #
    # def plot(self):
    #     dates = self.panel.dates.to_list()
    #     fig, axes = plt.subplots(3, 1, sharex=True)
    #     plots = (
    #         (axes[0], self.posterior, "posterior", "posterior probability"),
    #         (axes[1], self.prior, "prior", "prior probability"),
    #         (
    #             axes[2],
    #             self.posterior - self.prior,
    #             "posterior - prior",
    #             "probability difference",
    #         ),
    #     )
    #     for ax, values, label, ylabel in plots:
    #         ax.plot(dates, values, label=label, alpha=0.8, linewidth=1.2)
    #         ax.scatter(dates, values, alpha=0.35, s=8)
    #         ax.set_ylabel(ylabel)
    #         ax.legend()
    #     axes[2].axhline(0.0, color="black", alpha=0.4, linewidth=0.8)
    #     fig.autofmt_xdate()
    #     return axes

    def plot(self):
        dates = self.panel.dates.to_list()

        fig, axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(11, 7),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1, 1, 0.9]},
        )

        diff = self.posterior - self.prior

        plots = (
            (axes[0], self.posterior, "Posterior probability", "Probability"),
            (axes[1], self.prior, "Prior probability", "Probability"),
            (axes[2], diff, "Posterior − prior", "Difference"),
        )

        for ax, values, title, ylabel in plots:
            ax.plot(
                dates,
                values,
                linewidth=1.8,
                alpha=0.9,
            )

            ax.scatter(
                dates,
                values,
                s=10,
                alpha=0.35,
                linewidths=0,
            )

            ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Format first two panels as percentages
        axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        # Format difference panel in basis points
        axes[2].yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x * 10_000:.1f} bp")
        )

        axes[2].axhline(
            0.0,
            color="black",
            linewidth=0.8,
            alpha=0.5,
        )

        # Cleaner date axis
        axes[2].xaxis.set_major_locator(mdates.YearLocator())
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        fig.suptitle(
            "Prior vs Posterior Scenario Probabilities",
            fontsize=14,
            fontweight="bold",
        )

        return axes


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
        viewed = self.view_distribution(panel, as_of=panel.dates[-1])
        return viewed.panel.with_prob(viewed.posterior)

    def view_distribution(
        self, panel: ScenarioPanel, *, as_of: datetime
    ) -> ViewedDistribution:
        if panel.kind != "return":
            raise ValueError("Views must be applied to a simple-return panel")
        result = entropy_pooling_probs(
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
        return market.viewed_returns(self)


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
