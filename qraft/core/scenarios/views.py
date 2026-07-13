"""Scenario view scheduling and viewed distribution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias, runtime_checkable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from qraft.core.panel import ScenarioPanel, expand_posterior_to_parent
from qraft.core.probability.prob_vector import ProbVector
from qraft.core.scenarios.view_types import ViewDiagnostics
from qraft.utils.helpers import str_to_datetime
from qraft.utils.helpers import weighted_moments

DateLike = datetime | str


@dataclass(frozen=True)
class ViewedDistribution:
    """Scenario distribution after applying a view.

    Stores the scenario panel, prior and posterior probabilities, diagnostics,
    and metadata for the date on which the view was evaluated.
    """

    panel: ScenarioPanel
    posterior: ProbVector
    prior: ProbVector
    diagnostics: ViewDiagnostics
    as_of: datetime
    name: str | None = None

    def moments(self) -> dict[str, dict[str, float]]:
        """Return prior and posterior moments by asset.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping from asset name to prior/posterior mean and standard
            deviation under the stored probability vectors.
        """
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
        """Expand posterior probabilities to a parent scenario panel.

        Parameters
        ----------
        parent
            Parent scenario panel whose probability vector defines the full
            scenario space.

        Returns
        -------
        ProbVector
            Posterior probabilities aligned to the parent panel.
        """
        return expand_posterior_to_parent(self.posterior, parent.prob, 1)

    def plot(self, *, title: str | None = None):
        """Plot cumulative prior and posterior probabilities.

        Parameters
        ----------
        title
            Optional figure title. When omitted, the distribution name is
            appended to the default title if available.

        Returns
        -------
        numpy.ndarray
            Matplotlib axes used for the three subplots.
        """
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
            (axes[0], prior_cum, "Prior cumulative probability", "Probability"),
            (axes[1], posterior_cum, "Posterior cumulative probability", "Probability"),
            (axes[2], cum_diff, "Cumulative posterior - prior", "Difference"),
        )

        for ax, values, subplot_title, ylabel in plots:
            ax.plot(dates, values, linewidth=1.8, alpha=0.9)
            ax.scatter(dates, values, s=10, alpha=0.35, linewidths=0)

            ax.set_title(subplot_title, loc="left", fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axes[2].yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x * 10_000:.1f} bp")
        )

        axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axes[2].xaxis.set_major_locator(mdates.YearLocator())
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        title_text = title or "Prior vs Posterior Scenario Probabilities"
        if self.name is not None and title is None:
            title_text += f" - {self.name}"
        fig.suptitle(title_text, fontsize=14, fontweight="bold")

        return axes


@runtime_checkable
class ScenarioView(Protocol):
    """Protocol for objects that transform scenario probabilities."""

    def view_distribution(
        self, panel: ScenarioPanel, *, as_of: datetime
    ) -> ViewedDistribution:
        """Apply the view to a scenario panel.

        Parameters
        ----------
        panel
            Scenario panel to reweight.
        as_of
            Evaluation timestamp for the view.

        Returns
        -------
        ViewedDistribution
            Result containing prior and posterior probabilities plus diagnostics.
        """
        ...


@dataclass(frozen=True, slots=True)
class ViewWindow:
    """Time interval over which a scenario view is active."""

    start: datetime
    end: datetime
    views: ScenarioView
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate that the window bounds are ordered."""
        if self.end < self.start:
            raise ValueError("ViewWindow end must be on or after start")

    def contains(self, t: datetime) -> bool:
        """Return whether a timestamp falls inside the window."""
        return self.start <= t <= self.end


ViewInput: TypeAlias = (
    "tuple[DateLike, DateLike, ScenarioView] "
    "| tuple[DateLike, DateLike, ScenarioView, str] "
    "| ViewWindow"
)


@dataclass(frozen=True, slots=True)
class ViewState:
    """Collection of scheduled scenario view windows."""

    windows: tuple[ViewWindow, ...] = ()

    def active_window_at(self, t: datetime) -> ViewWindow | None:
        """Return the first active view window at a timestamp.

        Parameters
        ----------
        t
            Timestamp to evaluate.

        Returns
        -------
        ViewWindow | None
            Matching window, or ``None`` when no window is active.
        """
        active = [window for window in self.windows if window.contains(t)]
        return active[0] if active else None


def normalize_view_window(window: ViewInput) -> ViewWindow:
    """Convert tuple-style view input into a view window.

    Parameters
    ----------
    window
        Existing ``ViewWindow`` or tuple containing start, end, view, and
        optionally a name.

    Returns
    -------
    ViewWindow
        Normalized view window with datetime bounds.
    """
    if isinstance(window, ViewWindow):
        return window
    if len(window) == 3:
        start, end, views = window
        name = None
    elif len(window) == 4:
        start, end, views, name = window
    else:
        raise ValueError(
            "View input must be (start, end, views) or (start, end, views, name)"
        )
    if isinstance(start, str):
        start = str_to_datetime(start)
    if isinstance(end, str):
        end = str_to_datetime(end)
    return ViewWindow(start=start, end=end, views=views, name=name)


def validate_non_overlapping_windows(windows: tuple[ViewWindow, ...]) -> None:
    """Raise when adjacent sorted view windows overlap.

    Parameters
    ----------
    windows
        View windows ordered by start date.
    """
    for prev, current in zip(windows, windows[1:], strict=False):
        if current.start <= prev.end:
            raise ValueError(
                "View windows must not overlap; "
                f"{prev.start!r} to {prev.end!r} overlaps "
                f"{current.start!r} to {current.end!r}."
            )
