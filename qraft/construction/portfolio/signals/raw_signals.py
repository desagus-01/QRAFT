# %%
import logging
from dataclasses import dataclass
from typing import Literal

import polars as pl
from polars import DataFrame, Series

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    type: Literal["momentum", "value"]
    values: DataFrame


# Momentum
def ewma_signal(
    risk_drivers: DataFrame,
    dates: Series,
    half_life: str = "60d",
    drop_nulls: bool = True,
) -> Signal:
    ewm_df = risk_drivers.with_columns(
        pl.all().diff().ewm_mean_by(dates, half_life=half_life)
    )

    return Signal(type="momentum", values=ewm_df.drop_nulls() if drop_nulls else ewm_df)
