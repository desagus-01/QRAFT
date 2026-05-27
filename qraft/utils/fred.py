from io import BytesIO
from pathlib import Path

import polars as pl
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

RATES = {
    "USDOLLAR": "DFF",
    "EURO": "ECBESTRVOLWGTTRMDMNRT",
    "GBPOUND": "IUDSOIA",
    "JPYEN": "IRSTCB01JPM156N",
}


def resolve_fred_symbol(symbol: str) -> str:
    """Resolve a local rate alias to the FRED series id."""
    return RATES.get(symbol.upper(), symbol.upper())


def download_fred(
    symbol: str = "DFF",
    *,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pl.DataFrame:
    """
    Download a single FRED series and return it as a Polars DataFrame.

    The returned DataFrame has a `date` column and one value column named with the
    resolved FRED series id.
    """
    fred_symbol = resolve_fred_symbol(symbol)
    client = session or requests.Session()
    response = client.get(FRED_URL, params={"id": fred_symbol}, timeout=timeout)
    response.raise_for_status()

    return (
        pl.read_csv(
            BytesIO(response.content),
            try_parse_dates=True,
            null_values=".",
        )
        .rename({"observation_date": "date"})
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col(fred_symbol).cast(pl.Float64, strict=False),
        )
    )


def save_fred_csv(
    path: str | Path,
    symbol: str = "DFF",
    *,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> None:
    """Download a FRED series and save it as CSV."""
    download_fred(symbol, timeout=timeout, session=session).write_csv(path)
