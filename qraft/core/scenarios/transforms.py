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
    name: str | None = None

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

    def plot(self, *, title: str | None = None):
        dates = self.panel.dates.to_list()

        fig, axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(11, 7),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1, 1, 0.9]},
        )

        prior_cum = self.prior.cumsum()
        posterior_cum = self.posterior.cumsum()
        cum_diff = posterior_cum - prior_cum

        plots = (
            (
                axes[0],
                prior_cum,
                "Prior cumulative probability",
                "Probability",
            ),
            (
                axes[1],
                posterior_cum,
                "Posterior cumulative probability",
                "Probability",
            ),
            (axes[2], cum_diff, "Cumulative posterior − prior", "Difference"),
        )

        for ax, values, subplot_title, ylabel in plots:
            ax.plot(dates, values, linewidth=1.8, alpha=0.9)
            ax.scatter(dates, values, s=10, alpha=0.35, linewidths=0)

            ax.set_title(subplot_title, loc="left", fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Format cumulative probabilities as percent; differences as basis points.
        axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
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

        title_text = title or "Prior vs Posterior Scenario Probabilities"
        if self.name is not None and title is None:
            title_text += f" — {self.name}"
        fig.suptitle(title_text, fontsize=14, fontweight="bold")

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
