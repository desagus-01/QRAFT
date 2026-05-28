import logging

import numpy as np
import polars as pl
from polars.dataframe.frame import DataFrame

from scenarios.types import ProbVector
from time_series.preprocessing.types import (
    TransformDecision,
)
from time_series.transforms.deseason import (
    deterministic_seasonal_adjustment,
)
from time_series.transforms.detrend import (
    add_detrend_column,
    add_differenced_columns,
)
from time_series.transforms.inverses import (
    DifferenceInverseSpec,
    InverseSpec,
    PolynomialInverseSpec,
    SeasonalInverseSpec,
)

logger = logging.getLogger(__name__)


def _group_detrend_assets(
    decision: dict[str, TransformDecision],
) -> dict[tuple[str, int], list[str]]:
    """Group assets by (transform kind, order)."""
    by_group: dict[tuple[str, int], list[str]] = {}

    for asset, dec in decision.items():
        if dec.order is None:
            continue
        by_group.setdefault((dec.kind, dec.order), []).append(asset)

    return by_group


def _apply_grouped_detrend(
    data: DataFrame,
    prob: ProbVector,
    by_group: dict[tuple[str, int], list[str]],
) -> tuple[DataFrame, dict[str, InverseSpec]]:
    """Apply each grouped transform to the data and retain inverse specs."""
    inverse_specs: dict[str, InverseSpec] = {}

    for (transform, order), assets in by_group.items():
        if transform == "difference":
            # store anchors from the series BEFORE differencing
            anchors_by_asset = {
                asset: data.select(asset).to_series().tail(order).to_numpy()
                for asset in assets
            }

            data = add_differenced_columns(
                data=data,
                assets=assets,
                difference=order,
                keep_all=True,
            )

            for asset in assets:
                inverse_specs[asset] = DifferenceInverseSpec(
                    order=order,
                    initial_values=np.asarray(anchors_by_asset[asset], dtype=float),
                )
        elif transform == "polynomial":
            data, betas_by_order = add_detrend_column(
                original_data=data,
                prob=prob,
                assets=assets,
                polynomial_orders=[order],
            )

            beta = betas_by_order[order]

            for asset in assets:
                inverse_specs[asset] = PolynomialInverseSpec(
                    order=order, betas=np.asarray(beta[asset], dtype=float)
                )

        else:
            raise ValueError(f"Unknown transform '{transform}' for assets {assets}")

    return data, inverse_specs


def _select_and_rename_detrended_columns(
    data: DataFrame,
    decision: dict[str, TransformDecision],
) -> DataFrame:
    """Keep only date + transformed asset columns, renamed back to asset names."""
    transformed_cols: list[str] = []
    rename_map: dict[str, str] = {}

    for asset, dec in decision.items():
        if dec.order is None:
            continue

        if dec.kind == "difference":
            col = f"{asset}_diff_{dec.order}"
        elif dec.kind == "polynomial":
            col = f"{asset}_detrended_p{dec.order}"
        else:
            raise ValueError(f"Unknown transform '{dec.kind}' for asset '{asset}'")

        if col not in data.columns:
            raise ValueError(
                f"Expected transformed column '{col}' not found. "
                f"Available columns: {data.columns}"
            )

        transformed_cols.append(col)
        rename_map[col] = asset

    return data.select(["date", *transformed_cols]).rename(rename_map)


def apply_detrend(
    data: DataFrame,
    decision: dict[str, TransformDecision],
    prob: ProbVector,
) -> tuple[DataFrame, dict[str, InverseSpec]]:
    """Apply detrending decisions and return date + transformed asset columns."""
    if not decision:
        return data.select(["date"]), {}

    by_group = _group_detrend_assets(decision)
    if not by_group:
        return data.select(["date"]), {}

    transformed, inverse_specs = _apply_grouped_detrend(
        data=data, prob=prob, by_group=by_group
    )
    selected = _select_and_rename_detrended_columns(transformed, decision)
    return selected, inverse_specs


def apply_deseason(
    data: pl.DataFrame,
    decision: dict[str, list[tuple[str, float]]],
) -> tuple[pl.DataFrame, dict[str, InverseSpec]]:
    assets = [asset for asset, seasons in decision.items() if seasons]
    if not assets:
        return data.select(["date"]), {}

    out = data.select(["date"])
    inverse_specs: dict[str, InverseSpec] = {}

    for asset in assets:
        seasons = decision[asset]
        omega = [rad for _, rad in seasons]

        seas_adj_res = deterministic_seasonal_adjustment(
            data=data,
            asset=asset,
            frequency_radians=omega,
        )

        out = out.join(seas_adj_res.residuals, on="date", how="left")
        inverse_specs[asset] = SeasonalInverseSpec(terms=seas_adj_res.terms)

    return out, inverse_specs


def overwrite_with_transforms(
    base: pl.DataFrame,
    patch: pl.DataFrame,
    assets: list[str],
    suffix: str,
) -> pl.DataFrame:
    j = base.join(patch, on="date", how="left", suffix=suffix)

    exprs: list[pl.Expr] = []
    drop_cols: list[str] = []

    for a in assets:
        pa = f"{a}{suffix}"
        if pa in j.columns:
            exprs.append(pl.col(pa).alias(a))
            drop_cols.append(pa)
        else:
            exprs.append(pl.col(a))

    return j.with_columns(exprs).drop(drop_cols)
