import math
from typing import Literal

import polars as pl
from polars import DataFrame
from polars._typing import RankMethod
from signals.raw_signals import Signal

Distortion = Literal[
    "identity",
    "terciles",
    "median",
    "winsorize",
    "tanh",
    "arctan",
    "power",
]


def signal_smoothing(signal_df: DataFrame, half_life: int) -> DataFrame:
    return signal_df.with_columns(pl.all().ewm_mean(half_life=half_life, adjust=False))


def signal_scoring(signal_df: DataFrame, half_life: int) -> DataFrame:
    return signal_df.with_columns(
        (pl.all() - pl.all().ewm_mean(half_life=half_life, adjust=False))
        / pl.all().ewm_std(half_life=half_life, adjust=False)
    ).drop_nans()  # needed as first row will NAN due to st dev


def signal_ranking(
    signal_df: DataFrame,
    method: RankMethod = "average",
) -> DataFrame:
    cols = signal_df.columns
    n = len(cols)

    if n < 2:
        raise ValueError("Ranking needs at least two signal columns.")

    return (
        signal_df.with_columns(
            pl.concat_list(cols)
            .list.eval((2 * pl.element().rank(method=method) - (n + 1)) / (n - 1))
            .alias("_ranks")
        )
        .with_columns([pl.col("_ranks").list.get(i).alias(cols[i]) for i in range(n)])
        .drop("_ranks")
    )


def signal_distortion(
    ranked_df: DataFrame,
    kind: Distortion,
    *,
    winsor_limit: float = 0.8,
    strength: float = 3.0,
    power: float = 2.0,
) -> DataFrame:
    x = pl.all()

    if kind == "identity":
        expr = x

    elif kind == "terciles":
        expr = (
            pl.when(x > 1 / 3)
            .then(1.0)
            .when(x < -1 / 3)
            .then(-1.0)
            .otherwise(0.0)
            .name.keep()
        )

    elif kind == "median":
        expr = (
            pl.when(x > 0).then(1.0).when(x < 0).then(-1.0).otherwise(0.0).name.keep()
        )

    elif kind == "winsorize":
        expr = x.clip(-winsor_limit, winsor_limit)

    elif kind == "tanh":
        # Smooth S-shaped distortion.
        expr = (x * strength).tanh() / math.tanh(strength)

    elif kind == "arctan":
        # Smooth S-shaped distortion, slightly gentler than tanh.
        expr = (x * strength).arctan() / math.atan(strength)

    elif kind == "power":
        # power > 1 suppresses weak/middle ranks and keeps extremes.
        # power < 1 makes middle ranks stronger.
        expr = x.sign() * x.abs().pow(power)

    else:
        raise ValueError(f"Unknown distortion: {kind}")

    return ranked_df.with_columns(expr)


def process_signal(
    signal: Signal,
    smoothing_factor: int,
    scoring_factor: int,
    ranking_method: RankMethod,
    distortion_method: Distortion = "identity",
) -> DataFrame:
    signal_df = signal.values
    if signal.type != "momentum":  # momentum signals are naturally smoothed already
        signal_df = signal_smoothing(signal_df=signal_df, half_life=smoothing_factor)
    scored_signal = signal_scoring(signal_df=signal_df, half_life=scoring_factor)
    ranked_signal = signal_ranking(signal_df=scored_signal, method=ranking_method)
    return signal_distortion(ranked_df=ranked_signal, kind=distortion_method)
