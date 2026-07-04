import csv
import io
import logging
import os
import zipfile
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
import polars as pl
import requests
from dotenv import load_dotenv
from polars import DataFrame
from requests import Session
from requests.exceptions import HTTPError

_ = load_dotenv()
session = requests.Session()

BASE_URL = "https://api.tiingo.com"
TIINGO_API = os.getenv("TIINGO_API")


def get_zipfile_from_response(response: requests.Response) -> zipfile.ZipFile:
    """
    Validate a HTTP response and return a ZipFile object.

    Parameters
    ----------
    response : requests.Response
        HTTP response containing a zip archive as its content.

    Returns
    -------
    zipfile.ZipFile
        In-memory ZipFile for further extraction.

    Raises
    ------
    HTTPError
        If the response status is not successful.
    """
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


def get_buffer_from_zipfile(
    zipdata: zipfile.ZipFile,
    filename: str,
    encoding: str = "utf-8",
) -> io.StringIO:
    """
    Extract a text file from a ZipFile and return a StringIO buffer.

    Parameters
    ----------
    zipdata : zipfile.ZipFile
        Open zip archive.
    filename : str
        Name of the file within the archive to extract.
    encoding : str, optional
        Text encoding to use when decoding bytes (default: 'utf-8').

    Returns
    -------
    io.StringIO
        In-memory text buffer for the extracted file.
    """
    with zipdata.open(filename) as f:
        text = f.read().decode(encoding)
    return io.StringIO(text)


def get_tiingo_tickers(
    asset_types: Iterable[str] = ("Stock",),
) -> pl.DataFrame:
    """
    Download and parse Tiingo's supported tickers listing.

    Parameters
    ----------
    asset_types : Iterable[str], optional
        Iterable of assetType values to filter the listing on (default: ("Stock",)).

    Returns
    -------
    pl.DataFrame
        DataFrame of tickers filtered by asset type with parsed start/end dates.
    """
    listing_file_url = (
        "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    )

    response = session.get(listing_file_url, timeout=30)
    zipdata = get_zipfile_from_response(response)
    raw_csv = get_buffer_from_zipfile(zipdata, "supported_tickers.csv")
    reader = csv.DictReader(raw_csv)

    rows = list(reader)
    asset_types_set = set(asset_types)

    return pl.DataFrame(
        [row for row in rows if row.get("assetType") in asset_types_set]
    ).with_columns(
        [
            pl.when(pl.col("startDate").str.strip_chars() == "")
            .then(None)
            .otherwise(pl.col("startDate"))
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .alias("startDate"),
            pl.when(pl.col("endDate").str.strip_chars() == "")
            .then(None)
            .otherwise(pl.col("endDate"))
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .alias("endDate"),
        ]
    )


def _request(
    session: Session,
    method: Literal["GET", "POST"],
    url: str,
    headers: dict[str, str] | None = None,
    **kwargs,
):
    """
    Internal wrapper around requests to standardise headers and error handling.

    Raises HTTPError after logging if response is unsuccessful.
    """
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    resp = session.request(method, url, headers=merged_headers, **kwargs)
    try:
        resp.raise_for_status()
    except HTTPError:
        logging.error("HTTP %s: %s", resp.status_code, resp.text)
        raise
    return resp


def _get_url(ticker: str) -> str:
    """
    Construct Tiingo price endpoint URL for a ticker.
    """
    return f"{BASE_URL}/tiingo/daily/{ticker}/prices"


def get_single_price(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    columns: str | None = None,
    frequency: str = "daily",
) -> DataFrame:
    """
    Fetch price data for a single Tiingo ticker and return as DataFrame.

    Parameters
    ----------
    ticker : str
        Tiingo ticker symbol.
    start_date, end_date : str | None, optional
        Inclusive date strings in 'YYYY-MM-DD' format specifying the range.
    columns : str | None, optional
        Comma-separated column selection as supported by Tiingo API.
    frequency : str, optional
        Resampling frequency to request from Tiingo (default: 'daily').

    Returns
    -------
    DataFrame
        Polars DataFrame built from the Tiingo JSON price response.
    """
    url = _get_url(ticker)

    params: dict[str, Any] = {
        "token": TIINGO_API,
        "format": "json",
        "resampleFreq": frequency,
    }
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if columns:
        params["columns"] = columns

    data: list[dict[str, Any]] = _request(session, "GET", url, params=params).json()
    return DataFrame(data)


def get_ticker_prices(
    tickers: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    columns: str | None = None,
    frequency: str = "daily",
    return_clean: bool = True,
):
    """
    Fetch and concatenate prices for multiple tickers into a single DataFrame.

    Parameters
    ----------
    tickers : list[str]
        List of Tiingo tickers.
    start_date, end_date, columns, frequency : see :func:`get_single_price`.
    return_clean : bool, optional
        If True, select and rename a small set of common columns for downstream use.

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame of ticker prices with 'date' parsed to datetime.date.
    """
    dfs = []
    for t in tickers:
        out = get_single_price(
            t,
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            frequency=frequency,
        ).with_columns(pl.lit(t).alias("ticker"))

        dfs.append(out)

    df = pl.concat(dfs, how="vertical_relaxed")

    df = df.with_columns(
        pl.col("date")
        .str.to_datetime(time_zone="UTC", strict=False)
        .dt.date()
        .alias("date")
    )

    if return_clean:
        df = df.select(
            [
                "date",
                "ticker",
                "close",
                pl.col("adjClose").alias("adj_close"),
                pl.col("adjVolume").alias("adj_volume"),
            ]
        )

    return df


def plot_ticker_lines(
    df: pl.DataFrame,
    date_column: str = "date",
    value_column: str = "adj_close",
    ticker_column: str = "ticker",
):
    """
    Plot time series lines for each ticker in ``df`` and return figure objects.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing date, ticker and value columns.
    date_column : str, optional
        Column name containing dates (default: 'date').
    value_column : str, optional
        Column name containing the values to plot (default: 'adj_close').
    ticker_column : str, optional
        Column name containing ticker identifiers (default: 'ticker').

    Returns
    -------
    list[tuple]
        List of tuples (ticker, fig, ax) for each plotted ticker.
    """
    tickers = df.select(pl.col(ticker_column).unique()).to_series().to_list()
    figs_axes = []

    for ticker in tickers:
        ticker_df = df.filter(pl.col(ticker_column) == ticker).sort(date_column)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            ticker_df[date_column].to_list(),
            ticker_df[value_column].to_list(),
            linewidth=1.5,
        )

        ax.set_title(f"{ticker}")
        ax.set_ylabel(value_column)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        figs_axes.append((ticker, fig, ax))

    return figs_axes


#
# def get_tiingo_tickers(csv_path: str = "./data/supported_tickers.csv") -> pl.DataFrame:
#     return pl.read_csv(csv_path, try_parse_dates=True)
#


def previous_business_day(d: date) -> date:
    """
    Return the previous business day (Mon-Fri) before the given date.

    Parameters
    ----------
    d : date
        Reference date.

    Returns
    -------
    date
        Most recent business day strictly before ``d``.
    """
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def business_day_x_years(target_bd: date, years: int) -> date:
    """
    Compute a business day approximately ``years`` years prior to ``target_bd``.

    The function attempts to preserve the month/day where possible, falling
    back to February 28 for leap-year edge cases, and then steps back to the
    nearest preceding business day if the result falls on a weekend.

    Parameters
    ----------
    target_bd : date
        Reference business day date.
    years : int
        Number of years to step back.

    Returns
    -------
    date
        Business day approximately ``years`` earlier than ``target_bd``.
    """
    try:
        new_date = target_bd.replace(year=target_bd.year - years)
    except ValueError:
        new_date = target_bd.replace(year=target_bd.year - years, month=2, day=28)

    while new_date.weekday() >= 5:
        new_date -= timedelta(days=1)
    return new_date


def get_tiingo_dates(date: date, years_back: int) -> tuple[date, date]:
    """
    Convenience helper returning a start/end date covering ``years_back`` to
    the most recent business day prior to ``date``.

    Parameters
    ----------
    date : date
        Anchor date to compute the trailing window.
    years_back : int
        Number of years to go back from the most recent business day.

    Returns
    -------
    tuple[date, date]
        (start_date, end_date) where end_date is the previous business day
        before ``date`` and start_date is approximately ``years_back`` years earlier.
    """
    target_enddate = previous_business_day(date)
    target_startdate = business_day_x_years(target_bd=target_enddate, years=years_back)
    return target_startdate, target_enddate


def create_tiingo_universe(
    tiingo_ticker_info: pl.DataFrame,
    end_date: date,
    exchanges: list[str] = ["NASDAQ", "NYSE"],
) -> pl.DataFrame:
    """
    Filter the Tiingo tickers listing into a USD-equity universe for a given end date.

    Parameters
    ----------
    tiingo_ticker_info : pl.DataFrame
        DataFrame as returned by :func:`get_tiingo_tickers`.
    end_date : date
        Date to use when filtering tickers whose endDate equals this value.
    exchanges : list[str], optional
        List of exchanges to include (default: ['NASDAQ', 'NYSE']).

    Returns
    -------
    pl.DataFrame
        Filtered universe of tickers matching the criteria.
    """
    return tiingo_ticker_info.filter(
        (pl.col("priceCurrency") == "USD")
        & (pl.col("exchange").is_in(exchanges))
        & (pl.col("assetType") == "Stock")
        & pl.col("startDate").is_not_null()
        & (pl.col("endDate") == pl.lit(end_date))
    )


def sample_tickers_from_universe(
    universe_df: pl.DataFrame, target_start_date: date, n: int, seed: int
) -> list[str]:
    """
    Randomly sample tickers from the universe that have started trading by ``target_start_date``.

    Parameters
    ----------
    universe_df : pl.DataFrame
        Universe dataframe with 'ticker' and 'startDate' columns.
    target_start_date : date
        Cutoff start date: tickers with startDate <= target_start_date are eligible.
    n : int
        Number of tickers to sample.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list[str]
        Sampled ticker symbols.
    """
    return (
        universe_df.filter(pl.col("startDate") <= pl.lit(target_start_date))
        .select("ticker")
        .unique(maintain_order=True)
        .sample(n=n, with_replacement=False, shuffle=True, seed=seed)
        .get_column("ticker")
        .to_list()
    )


def _coerce_date(value: date | str) -> date:
    """
    Coerce a string in 'YYYY-MM-DD' format or a date into a date object.
    """
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_workflow_dates(
    *,
    as_of_date: date | str | None = None,
    years_back: int = 10,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> tuple[date, date, str, str]:
    """
    Resolve start/end dates for workflows either from explicit dates or from
    an anchor ``as_of_date`` and ``years_back`` window.

    Parameters
    ----------
    as_of_date : date | str | None, optional
        Anchor date from which to compute the trailing window (default: today).
    years_back : int, optional
        Years back to include in the window when ``start_date`` is not provided.
    start_date, end_date : date | str | None, optional
        If both provided, they are validated and used directly.

    Returns
    -------
    tuple[date, date, str, str]
        (target_startdate, target_enddate, start_date_str, end_date_str) where
        the string values are formatted 'YYYY-MM-DD' for API calls.
    """
    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be provided together.")

    if start_date is not None and end_date is not None:
        start_dt = _coerce_date(start_date)
        end_dt = _coerce_date(end_date)

        if start_dt > end_dt:
            raise ValueError("start_date must be on or before end_date.")

        target_enddate = previous_business_day(end_dt + timedelta(days=1))
        target_startdate = start_dt

        return (
            target_startdate,
            target_enddate,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )

    as_of = _coerce_date(as_of_date) if as_of_date is not None else date.today()
    target_startdate, target_enddate = get_tiingo_dates(as_of, years_back=years_back)

    return (
        target_startdate,
        target_enddate,
        target_startdate.strftime("%Y-%m-%d"),
        target_enddate.strftime("%Y-%m-%d"),
    )


def build_sampled_universe(
    target_startdate: date,
    target_enddate: date,
    exchanges: Iterable[str] = ("NASDAQ", "NYSE"),
    n_tickers: int = 100,
    seed: int = 1,
) -> list[str]:
    """
    Build a sampled universe of tickers with available history between dates.

    Parameters
    ----------
    target_startdate, target_enddate : date
        Window endpoints used to filter eligible tickers.
    exchanges : Iterable[str], optional
        Exchanges to consider (default: ('NASDAQ', 'NYSE')).
    n_tickers : int, optional
        Number of tickers to sample.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    list[str]
        List of sampled tickers.
    """
    tiingo_tickers = get_tiingo_tickers()
    universe_df = create_tiingo_universe(
        tiingo_ticker_info=tiingo_tickers,
        end_date=target_enddate,
        exchanges=list(exchanges),
    ).filter(pl.col("startDate") <= pl.lit(target_startdate))

    available = universe_df.select("ticker").unique().height
    if n_tickers > available:
        raise ValueError(f"Requested {n_tickers} tickers, only {available} available.")

    return sample_tickers_from_universe(
        universe_df=universe_df,
        target_start_date=target_startdate,
        n=n_tickers,
        seed=seed,
    )


def get_sampled_ticker_prices(
    *,
    as_of_date: date | str | None = None,
    years_back: int = 10,
    n_tickers: int = 100,
    seed: int = 1,
    exchanges: Iterable[str] = ("NASDAQ", "NYSE"),
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    frequency: str = "daily",
    columns: str | None = None,
) -> pl.DataFrame:
    """
    High-level helper to sample tickers and return their price series.

    Parameters
    ----------
    as_of_date, years_back, start_date, end_date : see :func:`resolve_workflow_dates`.
    n_tickers, seed, exchanges : see :func:`build_sampled_universe`.
    frequency, columns : forwarded to :func:`get_ticker_prices`.

    Returns
    -------
    pl.DataFrame
        Concatenated price data for sampled tickers.
    """
    (
        target_startdate,
        target_enddate,
        start_date_str,
        end_date_str,
    ) = resolve_workflow_dates(
        as_of_date=as_of_date,
        years_back=years_back,
        start_date=start_date,
        end_date=end_date,
    )

    sampled_tickers = build_sampled_universe(
        target_startdate=target_startdate,
        target_enddate=target_enddate,
        exchanges=exchanges,
        n_tickers=n_tickers,
        seed=seed,
    )

    prices = get_ticker_prices(
        tickers=sampled_tickers,
        start_date=start_date_str,
        end_date=end_date_str,
        columns=columns,
        frequency=frequency,
        return_clean=True,
    )

    return prices


def clean_and_save_sample(sample_df: pl.DataFrame, path_to_save: str) -> None:
    """
    Pivot and save a sampled price DataFrame to CSV with adjusted closes.

    Parameters
    ----------
    sample_df : pl.DataFrame
        DataFrame containing columns ['date', 'ticker', 'adj_close'].
    path_to_save : str
        File path where the resulting CSV should be written.
    """
    clean_df = (
        sample_df.select(["date", "ticker", "adj_close"])
        .pivot("ticker", index="date", values="adj_close")
        .sort("date")
    )

    clean_df.write_csv(path_to_save)


def import_tickers_and_factors(
    tickers_path: str, factors_path: str
) -> tuple[pl.DataFrame, list[str]]:
    data = pl.read_csv(tickers_path)
    factors = pl.read_csv(factors_path)
    factors_cols = [column for column in factors.columns if column != "date"]
    merged = factors.join(data, on="date", how="left")
    merged = merged.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d")).filter(
        pl.col("date") <= datetime(2026, 3, 19)
    )
    return merged, factors_cols
