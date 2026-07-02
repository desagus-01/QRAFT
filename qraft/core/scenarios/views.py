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
class ViewEvent:
    as_of: datetime
    views: ScenarioView


ViewInput: TypeAlias = "tuple[DateLike, ScenarioView] | ViewEvent"


@dataclass(frozen=True, slots=True)
class ViewState:
    events: tuple[ViewEvent, ...] = ()
    current: ScenarioView | None = None

    def latest_event_at(self, t: datetime) -> ViewEvent | None:
        active = [event for event in self.events if event.as_of <= t]
        return active[-1] if active else None

    @property
    def has_current_only_views(self) -> bool:
        return self.current is not None


def normalize_view_event(event: ViewInput) -> ViewEvent:
    if isinstance(event, ViewEvent):
        return event
    as_of, views = event
    if isinstance(as_of, str):
        as_of = str_to_datetime(as_of)
    return ViewEvent(as_of=as_of, views=views)
