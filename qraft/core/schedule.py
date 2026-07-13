"""Rebalance-calendar utilities for selecting decision and execution bars."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import polars as pl

Cadence = Literal["every_bar", "week_end", "month_end", "quarter_end", "year_end"]


@dataclass(frozen=True, slots=True)
class RebalanceSchedule:
    """Select decision bars from a trading calendar using a rebalance cadence."""

    cadence: Cadence = "quarter_end"

    def _choose_bucket(self) -> str:
        match self.cadence:
            case "month_end":
                return "1mo"
            case "week_end":
                return "1w"
            case "quarter_end":
                return "1q"
            case "year_end":
                return "1y"
            case "every_bar":
                raise ValueError("doesn't count")

    def decision_bars(self, trading_bars: list[datetime]) -> list[datetime]:
        """Return trading bars where allocation decisions should be made."""
        if self.cadence == "every_bar":
            return list(trading_bars)
        bucket = self._choose_bucket()
        return (
            pl.DataFrame({"date": trading_bars})
            .group_by(
                pl.col("date").dt.truncate(bucket).alias("bucket"),
                maintain_order=True,
            )
            .agg(pl.col("date").max().alias("bar"))
            .get_column("bar")
            .sort()
            .to_list()
        )

    @staticmethod
    def execution_bar(
        decision_bar: datetime, trading_bars: list[datetime]
    ) -> datetime | None:
        """Return the next trading bar after the decision, or ``None`` if last."""
        i = trading_bars.index(decision_bar)
        return trading_bars[i + 1] if i + 1 < len(trading_bars) else None

    def decision_steps(
        self, trading_bars: list[datetime]
    ) -> list[tuple[datetime, datetime]]:
        """(decision_bar, execution_bar) pairs; a final decision with no next bar is dropped."""
        steps: list[tuple[datetime, datetime]] = []
        for d in self.decision_bars(trading_bars):
            e = self.execution_bar(d, trading_bars)
            if e is not None:
                steps.append((d, e))
        return steps
