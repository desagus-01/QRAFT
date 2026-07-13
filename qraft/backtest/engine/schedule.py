from dataclasses import dataclass
from datetime import datetime
import logging

from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.core.snapshot import MarketSnapshot
from qraft.utils.log import info_event


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    index: int
    decision_bar: datetime
    execution_bar: datetime
    snapshot: MarketSnapshot


def decision_points(
    market: MarketData, schedule: RebalanceSchedule, warmup: int, step_size: int = 1
) -> list[DecisionPoint]:
    all_bars = market.trading_bars
    decision_bars = {d for d, _ in schedule.decision_steps(all_bars)}
    eligible_bars = [
        bar
        for i, bar in enumerate(all_bars)
        if bar in decision_bars and i + 1 >= warmup
    ]
    active_view_points = [
        (bar, window)
        for bar in eligible_bars
        if (window := market.views.active_window_at(bar)) is not None
    ]
    if active_view_points:
        info_event(
            logger,
            "views.decision_snapshots_pending",
            "Decision snapshots will apply scenario views",
            active_view_decisions=len(active_view_points),
            eligible_decisions=len(eligible_bars),
            views=tuple(dict.fromkeys(window.name for _, window in active_view_points)),
            first_date=active_view_points[0][0],
            last_date=active_view_points[-1][0],
        )
    points: list[DecisionPoint] = []

    for i, bar in enumerate(all_bars):
        if bar in decision_bars and i + 1 >= warmup:
            execution_bar = schedule.execution_bar(
                decision_bar=bar, trading_bars=all_bars
            )
            if execution_bar is None:
                continue
            points.append(
                DecisionPoint(
                    index=len(points),
                    decision_bar=bar,
                    execution_bar=execution_bar,
                    snapshot=market.snapshot_at(
                        t=bar, t_next=execution_bar, step_size=step_size
                    ),
                )
            )
    active = sum(1 for point in points if point.snapshot.applied_views is not None)
    if active:
        active_names = [
            point.snapshot.applied_views.name
            for point in points
            if point.snapshot.applied_views is not None
        ]
        first = next(
            point.decision_bar
            for point in points
            if point.snapshot.applied_views is not None
        )
        last = next(
            point.decision_bar
            for point in reversed(points)
            if point.snapshot.applied_views is not None
        )
        info_event(
            logger,
            "views.active_decision_snapshots",
            "Scenario views active for decision snapshots",
            active_decisions=active,
            decisions=len(points),
            views=tuple(dict.fromkeys(active_names)),
            first_date=first,
            last_date=last,
        )
    return points
