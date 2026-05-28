from typing import Literal

import numpy as np
from numpy.typing import NDArray

WEIGHT_MODE = Literal["buy_and_hold", "static"]


def validate_target_weights(
    target_weights: dict[str, float],
    asset_order: list[str],
) -> None:
    """Raise if *target_weights* are missing, extra, non-finite, negative,
    or do not sum to 1.
    """
    missing = [a for a in asset_order if a not in target_weights]
    extra = [a for a in target_weights if a not in asset_order]

    if missing:
        raise KeyError(f"Missing target weights for assets: {missing}")
    if extra:
        raise KeyError(f"Extra target weights for unknown assets: {extra}")

    w = np.array([target_weights[a] for a in asset_order], dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("target_weights must be finite")
    if np.any(w < 0):
        raise ValueError("target_weights must be non-negative")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"target_weights must sum to 1. Got {w.sum():.8f}")


def initial_portfolio_values_from_shares(
    prices: NDArray[np.floating],
    initial_asset_shares: dict[str, int],
    asset_order: list[str],
) -> NDArray[np.floating]:
    """Compute per-simulation t0 portfolio value from the pre-built price stack.

    Parameters
    ----------
    prices
        Price stack of shape (n_assets, n_sims, n_periods) where column 0 is t0.
    """
    missing = [a for a in asset_order if a not in initial_asset_shares]
    if missing:
        raise KeyError(f"No shares specified for assets: {missing}")

    shares = np.array([initial_asset_shares[a] for a in asset_order], dtype=float)
    if np.any(shares < 0):
        raise ValueError("initial_asset_shares must be non-negative")

    # t0 prices are at column 0; multiply by shares and sum over assets
    return (prices[:, :, 0] * shares[:, None]).sum(axis=0)  # (n_sims,)


def portfolio_values_static_weights(
    prices: NDArray[np.floating],
    target_weights: dict[str, float],
    asset_order: list[str],
    initial_portfolio_values: NDArray[np.floating],
    safe_eps: float = 1e-12,
) -> NDArray[np.floating]:
    """Compute portfolio values under static (rebalanced) weights.

    Parameters
    ----------
    prices
        Pre-built price stack of shape (n_assets, n_sims, n_periods) where
        column 0 is the known t0 price.
    """
    _, n_sims, n_periods = prices.shape

    initial_portfolio_values = np.asarray(initial_portfolio_values, dtype=float)
    if initial_portfolio_values.shape != (n_sims,):
        raise ValueError(
            f"initial_portfolio_values must have shape ({n_sims},), "
            f"got {initial_portfolio_values.shape}"
        )

    validate_target_weights(target_weights, asset_order)
    w = np.array([target_weights[a] for a in asset_order], dtype=float)

    prev = prices[:, :, :-1]
    curr = prices[:, :, 1:]

    denom = prev.astype(float, copy=True)
    denom[np.abs(denom) < safe_eps] = np.nan
    asset_returns = (curr - prev) / denom  # (n_assets, n_sims, n_periods - 1)

    portfolio_returns = np.einsum("a,ast->st", w, asset_returns)

    values = np.empty((n_sims, n_periods), dtype=float)
    values[:, 0] = initial_portfolio_values
    values[:, 1:] = initial_portfolio_values[:, None] * np.cumprod(
        1.0 + portfolio_returns,
        axis=1,
    )
    return values


def portfolio_weights_forecast_buy_and_hold(
    asset_values: NDArray[np.floating],
    portfolio_values: NDArray[np.floating],
    asset_order: list[str],
    safe_eps: float = 1e-12,
) -> dict[str, NDArray[np.floating]]:
    denom = portfolio_values.astype(float, copy=True)
    denom[np.abs(denom) < safe_eps] = np.nan

    weights_arr = asset_values / denom[None, :, :]
    return {asset: weights_arr[i] for i, asset in enumerate(asset_order)}


def portfolio_weights_forecast_static(
    target_weights: dict[str, float],
    n_sims: int,
    n_periods: int,
    asset_order: list[str],
) -> dict[str, NDArray[np.floating]]:
    return {
        asset: np.full((n_sims, n_periods), float(target_weights[asset]), dtype=float)
        for asset in asset_order
    }
