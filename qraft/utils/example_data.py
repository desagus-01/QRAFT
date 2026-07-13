from datetime import datetime

import numpy as np
import polars as pl

from qraft import AssetUniverse, MarketData, Prior


def synthetic_vix_market() -> MarketData:
    """Build the shared synthetic market used by the VIX view examples."""
    rng = np.random.default_rng(11)
    n_days = 540
    dates = pl.date_range(
        datetime(2021, 1, 4),
        datetime(2021, 1, 4) + pl.duration(days=n_days - 1),
        interval="1d",
        eager=True,
    ).cast(pl.Datetime)

    vix = np.empty(n_days)
    spy = np.empty(n_days)
    tlt = np.empty(n_days)
    gld = np.empty(n_days)
    vix[0] = 18.0
    spy[0], tlt[0], gld[0] = 100.0, 100.0, 100.0

    for t in range(1, n_days):
        regime_anchor = 15.0 if t < 180 else 27.0 if t < 360 else 18.0
        vix[t] = np.clip(
            0.92 * vix[t - 1] + 0.08 * regime_anchor + rng.normal(0.0, 1.5),
            10,
            50,
        )
        vix_change = (vix[t] - vix[t - 1]) / 100.0

        spy[t] = spy[t - 1] * (
            1.0 + 0.00050 - 1.6 * vix_change + rng.normal(0.0, 0.010)
        )
        tlt[t] = tlt[t - 1] * (
            1.0 + 0.00010 + 0.9 * vix_change + rng.normal(0.0, 0.006)
        )
        gld[t] = gld[t - 1] * (
            1.0 + 0.00020 + 0.5 * vix_change + rng.normal(0.0, 0.007)
        )

    prices = pl.DataFrame(
        {
            "date": dates,
            "SPY": spy,
            "TLT": tlt,
            "GLD": gld,
            "VIX": vix,
        }
    )

    universe = AssetUniverse(assets=["SPY", "TLT", "GLD"], factors=["VIX"])
    return MarketData.from_prices(
        prices,
        universe,
        prior=Prior.time_conditioned(half_life=63),
    )
