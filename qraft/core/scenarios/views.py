from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias

from qraft.core.panel import ScenarioPanel
from qraft.utils.helpers import str_to_datetime

DateLike = datetime | str


class ScenarioView(Protocol):
    def apply(self, panel: ScenarioPanel) -> ScenarioPanel: ...


@dataclass(frozen=True, slots=True)
class ViewWindow:
    start: datetime
    end: datetime
    views: ScenarioView
    name: str | None = None

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("ViewWindow end must be on or after start")

    def contains(self, t: datetime) -> bool:
        return self.start <= t <= self.end


ViewInput: TypeAlias = (
    "tuple[DateLike, DateLike, ScenarioView] "
    "| tuple[DateLike, DateLike, ScenarioView, str] "
    "| ViewWindow"
)


@dataclass(frozen=True, slots=True)
class ViewState:
    windows: tuple[ViewWindow, ...] = ()

    def active_window_at(self, t: datetime) -> ViewWindow | None:
        active = [window for window in self.windows if window.contains(t)]
        return active[0] if active else None


def normalize_view_window(window: ViewInput) -> ViewWindow:
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
    for prev, current in zip(windows, windows[1:], strict=False):
        if current.start <= prev.end:
            raise ValueError(
                "View windows must not overlap; "
                f"{prev.start!r} to {prev.end!r} overlaps "
                f"{current.start!r} to {current.end!r}."
            )
