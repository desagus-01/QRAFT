from datetime import datetime

import numpy as np
import polars as pl

from qraft import AssetUniverse, MarketData, Prior


def synthetic_market_example() -> MarketData:
    """Build the shared synthetic market used by the examples."""
    rng = np.random.default_rng(11)
    n_days = 540
    dates = pl.date_range(
        datetime(2021, 1, 4),
        datetime(2021, 1, 4) + pl.duration(days=n_days - 1),
        interval="1d",
        eager=True,
    ).cast(pl.Datetime)

    stress = np.empty(n_days)
    growth = np.empty(n_days)
    rates = np.empty(n_days)
    nova = np.empty(n_days)
    orion = np.empty(n_days)
    terra = np.empty(n_days)
    lumen = np.empty(n_days)
    cypher = np.empty(n_days)
    stress[0] = 18.0
    growth[0] = 100.0
    rates[0] = 100.0
    nova[0], orion[0], terra[0], lumen[0], cypher[0] = (
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
    )

    for t in range(1, n_days):
        stress_anchor = 15.0 if t < 180 else 27.0 if t < 360 else 18.0
        growth_anchor = 104.0 if t < 180 else 96.0 if t < 360 else 101.0
        rate_anchor = 98.0 if t < 220 else 106.0 if t < 420 else 101.0
        stress[t] = np.clip(
            0.92 * stress[t - 1] + 0.08 * stress_anchor + rng.normal(0.0, 1.5),
            10,
            50,
        )
        growth[t] = np.clip(
            0.96 * growth[t - 1] + 0.04 * growth_anchor + rng.normal(0.0, 1.0),
            80,
            120,
        )
        rates[t] = np.clip(
            0.97 * rates[t - 1] + 0.03 * rate_anchor + rng.normal(0.0, 0.8),
            85,
            115,
        )
        stress_change = (stress[t] - stress[t - 1]) / 100.0
        growth_change = (growth[t] - growth[t - 1]) / 100.0
        rate_change = (rates[t] - rates[t - 1]) / 100.0

        nova[t] = nova[t - 1] * (
            1.0
            + 0.00050
            - 1.4 * stress_change
            + 0.7 * growth_change
            + rng.normal(0.0, 0.010)
        )
        orion[t] = orion[t - 1] * (
            1.0
            + 0.00010
            + 0.8 * stress_change
            - 0.6 * rate_change
            + rng.normal(0.0, 0.006)
        )
        terra[t] = terra[t - 1] * (
            1.0
            + 0.00020
            + 0.4 * stress_change
            - 0.2 * growth_change
            + rng.normal(0.0, 0.007)
        )
        lumen[t] = lumen[t - 1] * (
            1.0
            + 0.00045
            - 1.1 * stress_change
            + 1.0 * growth_change
            - 0.4 * rate_change
            + rng.normal(0.0, 0.012)
        )
        cypher[t] = cypher[t - 1] * (
            1.0
            + 0.00030
            - 0.5 * stress_change
            + 0.3 * growth_change
            + 0.2 * rate_change
            + rng.normal(0.0, 0.008)
        )

    prices = pl.DataFrame(
        {
            "date": dates,
            "NOVA": nova,
            "ORION": orion,
            "TERRA": terra,
            "LUMEN": lumen,
            "CYPHER": cypher,
            "STRESS_INDEX": stress,
            "GROWTH_PULSE": growth,
            "RATE_WAVE": rates,
        }
    )

    universe = AssetUniverse(
        assets=["NOVA", "ORION", "TERRA", "LUMEN", "CYPHER"],
        factors=["STRESS_INDEX", "GROWTH_PULSE", "RATE_WAVE"],
    )
    return MarketData.from_prices(
        prices,
        universe,
        prior=Prior.time_conditioned(half_life=63),
    )
