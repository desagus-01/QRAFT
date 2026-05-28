from dataclasses import dataclass

import polars as pl
from utils.helpers import wide_to_long


@dataclass(frozen=True, slots=True)
class PortfolioInfoT0:
    all_info: pl.DataFrame
    estimated_t0: float
    shares_mapping: dict[str, int]
    t0_prices: dict[str, float]


def get_latest_prices(data_long: pl.DataFrame) -> pl.DataFrame:
    required = {"ticker", "adj_close"}
    missing = required - set(data_long.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if "date" in data_long.columns:
        data_long = data_long.with_columns(pl.col("date").cast(pl.Date)).sort(
            ["ticker", "date"]
        )
    else:
        dupes = (
            data_long.group_by("ticker")
            .len()
            .filter(pl.col("len") > 1)
            .select("ticker")
            .to_series()
            .to_list()
        )
        if dupes:
            raise ValueError(
                "Multiple rows per ticker found but no 'date' column is present. "
                f"Ambiguous latest price for tickers: {dupes}"
            )

    latest_price = data_long.group_by("ticker").agg(
        pl.col("adj_close").last().alias("adj_close")
    )

    if latest_price.height == 0:
        raise ValueError("No prices available after preprocessing")

    return latest_price


def equal_weight_target_weights(asset_order: list[str]) -> dict[str, float]:
    if not asset_order:
        raise ValueError("asset_order cannot be empty")

    w = 1.0 / len(asset_order)
    return {asset: w for asset in asset_order}


def equal_weight_shares_from_prices(
    latest_price: pl.DataFrame,
    initial_value: float,
    n_assets: int | None = None,
) -> pl.DataFrame:
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")

    if n_assets is None:
        n_assets = latest_price.height

    if n_assets <= 0:
        raise ValueError("n_assets must be positive")

    if latest_price.height == 0:
        raise ValueError("latest_price cannot be empty")

    bad_prices = latest_price.filter(pl.col("adj_close") <= 0)
    if bad_prices.height > 0:
        raise ValueError("All adj_close values must be positive")

    assigned = initial_value / n_assets

    return (
        latest_price.with_columns(
            (pl.lit(assigned) / pl.col("adj_close"))
            .floor()
            .cast(pl.Int64)
            .alias("shares")
        )
        .with_columns((pl.col("shares") * pl.col("adj_close")).alias("value_allocated"))
        .with_columns(
            (pl.col("value_allocated") / pl.lit(initial_value)).alias(
                "pct_of_portfolio"
            ),
            pl.lit(1.0 / n_assets).alias("target_pct"),
        )
        .select(
            [
                "ticker",
                "adj_close",
                "shares",
                "value_allocated",
                "pct_of_portfolio",
                "target_pct",
            ]
        )
    )


def portfolio_value(positions: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    required = {"adj_close", "shares"}
    if "value_allocated" not in positions.columns:
        missing = required - set(positions.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")
        positions = positions.with_columns(
            (pl.col("adj_close") * pl.col("shares")).alias("value_allocated")
        )

    total = float(positions["value_allocated"].sum())

    if total == 0:
        positions = positions.with_columns(pl.lit(0.0).alias("pct_of_portfolio"))
    else:
        positions = positions.with_columns(
            (pl.col("value_allocated") / pl.lit(total)).alias("pct_of_portfolio")
        )

    return total, positions


def build_equal_weight_portfolio_from_df(
    data: pl.DataFrame,
    initial_value: float,
    assets: list[str] | None = None,
) -> PortfolioInfoT0:
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")

    cols = set(data.columns)

    if {"ticker", "adj_close"}.issubset(cols):
        data_long = (
            data.with_columns(pl.col("date").cast(pl.Date)) if "date" in cols else data
        )

        if assets is None:
            assets = data_long.select("ticker").unique().to_series().to_list()
        else:
            data_long = data_long.filter(pl.col("ticker").is_in(assets))
    else:
        if assets is None:
            assets = [c for c in data.columns if c != "date"]
        data_long = wide_to_long(data, assets)

    if not assets:
        raise ValueError("No assets provided or inferred")

    latest_price = get_latest_prices(data_long)

    available_assets = set(latest_price["ticker"].to_list())
    missing_assets = [a for a in assets if a not in available_assets]
    if missing_assets:
        raise KeyError(f"Missing latest prices for assets: {missing_assets}")

    latest_price = latest_price.filter(pl.col("ticker").is_in(assets))

    eq_portfolio = equal_weight_shares_from_prices(
        latest_price=latest_price,
        initial_value=initial_value,
        n_assets=len(assets),
    )
    total, positions = portfolio_value(eq_portfolio)

    shares_mapping = dict(
        zip(
            positions["ticker"].to_list(),
            positions["shares"].to_list(),
        )
    )

    t0_prices = dict(
        zip(
            latest_price["ticker"].to_list(),
            latest_price["adj_close"].to_list(),
        )
    )

    return PortfolioInfoT0(
        all_info=positions,
        estimated_t0=total,
        shares_mapping=shares_mapping,
        t0_prices=t0_prices,
    )
