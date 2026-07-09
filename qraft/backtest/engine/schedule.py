from dataclasses import dataclass
from datetime import datetime

from qraft.core.market import MarketData
from qraft.core.schedule import RebalanceSchedule
from qraft.core.snapshot import MarketSnapshot


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
    return points
