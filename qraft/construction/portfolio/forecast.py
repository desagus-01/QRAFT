from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame
from portfolio.positions import (
    WEIGHT_MODE,
    initial_portfolio_values_from_shares,
    portfolio_values_static_weights,
    portfolio_weights_forecast_buy_and_hold,
    portfolio_weights_forecast_static,
    validate_target_weights,
)
from scenarios.panel import ScenarioPanel
from scenarios.types import ProbVector, validate_prob_vector
from utils.visuals import plot_simulation_results

PnL_OPTIONS = Literal["relative", "absolute", "log"]


PortfolioPanelKind = Literal[
    "value",
    "incremental_pnl",
    "cumulative_performance",
    "loss",
]


@dataclass(frozen=True, slots=True)
class PortfolioForecast:
    """
    Full Monte Carlo portfolio forecast.

    Shape conventions
    -----------------
    values:
        Shape (n_paths, n_periods). Includes t0 in column 0.

    incremental_pnl:
        Shape (n_paths, n_periods - 1) in the normal case.
        Each column is the incremental return/PnL from one period to the next.

    cumulative_performance:
        Derived from incremental_pnl.
        Shape (n_paths, n_periods). Column 0 is always zero.

    path_probs:
        One probability per simulated path.

    asset_weights:
        Dict asset -> array of shape (n_paths, n_periods).
        Includes t0 weights in column 0.
    """

    values: NDArray[np.floating]
    incremental_pnl: NDArray[np.floating]
    pnl_type: PnL_OPTIONS
    weight_mode: WEIGHT_MODE
    asset_weights: dict[str, NDArray[np.floating]]
    path_probs: ProbVector

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        incremental_pnl = np.asarray(self.incremental_pnl, dtype=float)

        if values.ndim != 2:
            raise ValueError(
                f"values must be 2-D (n_paths, n_periods); got shape={values.shape}"
            )

        if incremental_pnl.ndim != 2:
            raise ValueError(
                "incremental_pnl must be 2-D "
                f"(n_paths, n_periods - 1); got shape={incremental_pnl.shape}"
            )

        n_paths, n_periods = values.shape

        if incremental_pnl.shape[0] != n_paths:
            raise ValueError(
                f"Path mismatch: values has {n_paths} paths but "
                f"incremental_pnl has {incremental_pnl.shape[0]}"
            )

        if incremental_pnl.shape[1] != n_periods - 1:
            raise ValueError(
                "incremental_pnl should have one fewer period than values. "
                f"values has {n_periods} periods, incremental_pnl has "
                f"{incremental_pnl.shape[1]} periods."
            )

        if self.pnl_type not in {"relative", "absolute", "log"}:
            raise ValueError(f"Unknown pnl_type: {self.pnl_type}")

        validate_prob_vector(self.path_probs)

        if len(self.path_probs) != n_paths:
            raise ValueError(
                f"path_probs length ({len(self.path_probs)}) must equal "
                f"number of paths ({n_paths})"
            )

        clean_weights: dict[str, NDArray[np.floating]] = {}
        for asset, arr in self.asset_weights.items():
            weight = np.asarray(arr, dtype=float)

            if weight.ndim != 2:
                raise ValueError(
                    f"Weight array for asset '{asset}' must be 2-D "
                    f"(n_paths, n_periods); got shape={weight.shape}"
                )

            if weight.shape != values.shape:
                raise ValueError(
                    f"Weight array shape mismatch for asset '{asset}': "
                    f"expected {values.shape}, got {weight.shape}"
                )

            clean_weights[asset] = weight

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "incremental_pnl", incremental_pnl)
        object.__setattr__(self, "asset_weights", clean_weights)

    @property
    def n_paths(self) -> int:
        return self.values.shape[0]

    @property
    def n_periods(self) -> int:
        """
        Number of portfolio value columns, including t0.
        """
        return self.values.shape[1]

    @property
    def n_horizons(self) -> int:
        """
        Number of forecast horizons, excluding t0.
        """
        return self.n_periods - 1

    @property
    def asset_names(self) -> list[str]:
        return list(self.asset_weights.keys())

    @property
    def cumulative_performance(self) -> NDArray[np.floating]:
        return cumulative_pnl(
            self.incremental_pnl,
            self.pnl_type,
        )

    @property
    def losses(self) -> NDArray[np.floating]:
        return -self.cumulative_performance

    def _validate_period(self, period: int) -> None:
        """
        Validate period using portfolio convention:

        period=0:
            t0 / initial portfolio state

        period=1:
            first forecast step

        period=n:
            nth forecast step
        """
        if period < 0:
            raise ValueError("period must be >= 0")

        if period > self.n_horizons:
            raise ValueError(
                f"period={period} is out of range. Valid range is 0..{self.n_horizons}."
            )

    def performance_at_period(self, period: int) -> NDArray[np.floating]:
        self._validate_period(period)
        return self.cumulative_performance[:, period]

    def loss_at_period(self, period: int) -> NDArray[np.floating]:
        self._validate_period(period)
        return self.losses[:, period]

    def value_at_period(self, period: int) -> NDArray[np.floating]:
        self._validate_period(period)
        return self.values[:, period]

    def at_period(self, period: int) -> ScenarioPanel:
        """
        Return one portfolio period as a ScenarioPanel.

        period=0 is t0. period=1 is the first forecast step.
        Rows are simulated paths. Columns are portfolio-level quantities.
        """
        self._validate_period(period)

        columns: dict[str, NDArray[np.floating]] = {
            "portfolio_value": self.values[:, period],
            "cumulative_performance": self.cumulative_performance[:, period],
            "loss": self.losses[:, period],
        }

        if period > 0:
            columns["incremental_pnl"] = self.incremental_pnl[:, period - 1]

        return ScenarioPanel(
            values=DataFrame(columns),
            dates=None,
            prob=self.path_probs,
        )

    def at_step(self, step: int) -> ScenarioPanel:
        """
        Return one forecast step as a ScenarioPanel.

        step is 1-based: step=1 is the first forecast step. t0 is excluded.
        Delegates to at_period(step).
        """
        if step < 1:
            raise ValueError("step must be >= 1")
        return self.at_period(step)

    def weights_at_period(self, period: int) -> ScenarioPanel:
        """
        Return portfolio asset weights at one period as a ScenarioPanel.

        period=0 is t0. period=1 is the first forecast step.
        Rows are simulated paths. Columns are assets.
        """
        self._validate_period(period)

        return ScenarioPanel(
            values=DataFrame(
                {
                    asset: weights[:, period]
                    for asset, weights in self.asset_weights.items()
                }
            ),
            dates=None,
            prob=self.path_probs,
        )

    def panel(
        self,
        kind: PortfolioPanelKind,
    ) -> ScenarioPanel:
        """
        Return a wide ScenarioPanel with one column per horizon.

        h0 is t0. h1 is the first forecast step.
        """
        if kind == "value":
            data = self.values

        elif kind == "incremental_pnl":
            data = self.incremental_pnl
            return ScenarioPanel(
                values=DataFrame(
                    {f"h{h}": data[:, h - 1] for h in range(1, self.n_horizons + 1)}
                ),
                dates=None,
                prob=self.path_probs,
            )

        elif kind == "cumulative_performance":
            data = self.cumulative_performance

        elif kind == "loss":
            data = self.losses

        else:
            raise ValueError(f"Unknown panel kind: {kind}")

        return ScenarioPanel(
            values=DataFrame({f"h{h}": data[:, h] for h in range(self.n_periods)}),
            dates=None,
            prob=self.path_probs,
        )

    def loss_panel(self) -> ScenarioPanel:
        return self.panel("loss")

    def cumulative_performance_panel(self) -> ScenarioPanel:
        return self.panel("cumulative_performance")

    def value_panel(self) -> ScenarioPanel:
        return self.panel("value")

    def plot(
        self,
        end_horizon: int | None = None,
        value_type: Literal["value", "pnl", "loss"] = "pnl",
        plot_cumulative: bool = False,
    ) -> None:
        if end_horizon is None:
            end_horizon = self.n_horizons

        self._validate_period(end_horizon)

        if value_type not in {"value", "pnl", "loss"}:
            raise ValueError(f"Unknown value_type: {value_type}")

        if value_type == "value":
            if plot_cumulative:
                initial = self.values[:, :1].astype(float, copy=True)
                initial[np.abs(initial) < 1e-12] = np.nan
                data = self.values[:, : end_horizon + 1] / initial - 1.0
                title = f"Portfolio Cumulative Value ({self.weight_mode})"
            else:
                data = self.values[:, : end_horizon + 1]
                title = f"Portfolio Value ({self.weight_mode})"

        elif value_type == "loss":
            data = self.losses[:, : end_horizon + 1]
            title = f"Portfolio Loss ({self.weight_mode})"

        else:
            if plot_cumulative:
                data = self.cumulative_performance[:, : end_horizon + 1]
                title = f"Portfolio Cumulative PnL ({self.weight_mode})"
            else:
                data = self.incremental_pnl[:, :end_horizon]
                title = f"Portfolio Incremental PnL ({self.weight_mode})"

        plot_simulation_results(data, title=title)


def cumulative_pnl(
    pnl: NDArray[np.floating],
    pnl_type: PnL_OPTIONS,
    end_horizon: int | None = None,
    at_horizon: int | None = None,
) -> NDArray[np.floating]:
    if end_horizon is not None and at_horizon is not None:
        raise ValueError("end_horizon and at_horizon are mutually exclusive")
    if end_horizon is not None and end_horizon <= 0:
        raise ValueError("end_horizon must be a positive integer")
    if at_horizon is not None and at_horizon <= 0:
        raise ValueError("at_horizon must be a positive integer")

    if pnl_type not in {"relative", "absolute", "log"}:
        raise ValueError(f"Unknown pnl_type: {pnl_type}")

    if pnl.ndim != 2:
        raise ValueError(
            f"pnl must be 2-D (n_paths, n_horizons); got shape {np.asarray(pnl).shape}"
        )

    # When at_horizon is set we only need to accumulate up to that point.
    effective_end = at_horizon if at_horizon is not None else end_horizon
    slice_ = slice(None, effective_end)
    p = pnl[:, slice_]
    zeros = np.zeros((p.shape[0], 1), dtype=float)

    if pnl_type == "relative":
        running = np.cumprod(1.0 + p, axis=1) - 1.0
    elif pnl_type == "absolute":
        running = np.cumsum(p, axis=1)
    else:  # log
        running = np.exp(np.cumsum(p, axis=1)) - 1.0

    full = np.concatenate([zeros, running], axis=1)  # (n_sims, H + 1)

    if at_horizon is not None:
        return full[:, at_horizon]  # (n_sims,)

    return full


def _validate_asset_order(
    asset_forecasts: dict[str, NDArray[np.floating]],
    asset_order: list[str],
) -> None:
    if not asset_order:
        raise ValueError("asset_order cannot be empty")

    missing = [a for a in asset_order if a not in asset_forecasts]
    extra = [a for a in asset_forecasts if a not in asset_order]

    if missing:
        raise KeyError(f"Missing asset forecasts for: {missing}")
    if extra:
        raise KeyError(f"Found extra asset forecasts not in asset_order: {extra}")


def _build_price_stack(
    asset_forecasts: dict[str, NDArray[np.floating]],
    asset_order: list[str],
) -> NDArray[np.floating]:
    _validate_asset_order(asset_forecasts, asset_order)

    arrays: list[NDArray[np.floating]] = []
    base_shape: tuple[int, int] | None = None

    for asset in asset_order:
        arr = np.asarray(asset_forecasts[asset], dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"Asset {asset} must have shape (n_sims, n_periods), got {arr.shape}"
            )

        if base_shape is None:
            base_shape = arr.shape
        elif arr.shape != base_shape:
            raise ValueError(
                f"Shape mismatch for asset {asset}: {arr.shape} != {base_shape}"
            )

        arrays.append(arr)

    return np.stack(arrays, axis=0)  # (n_assets, n_sims, n_periods)


def _shares_vector(
    initial_asset_shares: dict[str, int],
    asset_order: list[str],
) -> NDArray[np.floating]:
    missing = [a for a in asset_order if a not in initial_asset_shares]
    if missing:
        raise KeyError(f"No shares specified for assets: {missing}")

    shares = np.array([initial_asset_shares[a] for a in asset_order], dtype=float)
    if np.any(shares < 0):
        raise ValueError("initial_asset_shares must be non-negative")

    return shares


def _build_asset_values_array(
    asset_forecasts: dict[str, NDArray[np.floating]],
    initial_asset_shares: dict[str, int],
    asset_order: list[str],
) -> NDArray[np.floating]:
    prices = _build_price_stack(asset_forecasts, asset_order)
    shares = _shares_vector(initial_asset_shares, asset_order)
    return prices * shares[:, None, None]  # (n_assets, n_sims, n_periods)


def _prepend_t0_prices(
    prices: NDArray[np.floating],
    asset_order: list[str],
    initial_prices: dict[str, float],
) -> NDArray[np.floating]:
    """Prepend last known prices (t0) to a price stack (n_assets, n_sims, n_periods).

    Returns a new writeable array of shape (n_assets, n_sims, n_periods + 1) where
    column 0 is the known t0 price (identical across all simulations) and columns
    1..n_periods are the simulated forecast prices.
    """
    missing = [a for a in asset_order if a not in initial_prices]
    if missing:
        raise KeyError(f"Missing t0 prices for assets: {missing}")

    t0 = np.array([initial_prices[a] for a in asset_order], dtype=float)
    n_assets, n_sims, _ = prices.shape
    t0_col = np.tile(t0[:, None, None], (1, n_sims, 1))
    return np.concatenate([t0_col, prices], axis=2)


def portfolio_value_forecast(
    asset_forecasts: dict[str, NDArray[np.floating]],
    initial_asset_shares: dict[str, int],
    asset_order: list[str] | None = None,
) -> NDArray[np.floating]:
    if asset_order is None:
        asset_order = list(initial_asset_shares.keys())

    asset_values = _build_asset_values_array(
        asset_forecasts=asset_forecasts,
        initial_asset_shares=initial_asset_shares,
        asset_order=asset_order,
    )
    return asset_values.sum(axis=0)


def pnl_from_values(
    values: NDArray[np.floating],
    mode: PnL_OPTIONS = "relative",
) -> NDArray[np.floating]:
    if mode not in {"relative", "absolute", "log"}:
        raise ValueError(f"Unknown mode: {mode!r}")

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"values must be 2-D (n_paths, n_periods); got shape {values.shape}"
        )

    prev = values[:, :-1]
    curr = values[:, 1:]

    if mode == "absolute":
        return curr - prev

    if mode == "relative":
        denom = prev.astype(float, copy=True)
        return (curr - prev) / denom

    denom = prev.astype(float, copy=True)
    ratio = curr / denom
    return np.log(ratio)


def portfolio_forecast(
    asset_forecasts: dict[str, NDArray[np.floating]],
    path_probs: ProbVector,
    initial_asset_shares: dict[str, int],
    initial_prices: dict[str, float],
    asset_order: list[str] | None = None,
    pnl_type: PnL_OPTIONS = "relative",
    weight_mode: WEIGHT_MODE = "buy_and_hold",
    target_weights: dict[str, float] | None = None,
    safe_eps: float = 1e-12,
) -> PortfolioForecast:
    """
    Build a full PortfolioForecast.

    Parameters
    ----------
    initial_prices
        Last known prices for each asset (t0). Used to anchor the price stack
        so that PnL and weights include the t0 -> t1 step.
    weight_mode
        - "buy_and_hold": fixed shares, drifting weights
        - "static": fixed target weights, implicitly rebalanced each step
    target_weights
        Required when weight_mode="static". Should map asset -> weight and sum to 1.
    """
    if asset_order is None:
        asset_order = list(initial_asset_shares.keys())

    prices = _build_price_stack(asset_forecasts, asset_order)
    prices = _prepend_t0_prices(prices, asset_order, initial_prices)
    _, n_sims, n_periods = prices.shape
    if weight_mode == "buy_and_hold":
        asset_values = (
            prices * _shares_vector(initial_asset_shares, asset_order)[:, None, None]
        )
        values = asset_values.sum(axis=0)
        weights = portfolio_weights_forecast_buy_and_hold(
            asset_values=asset_values,
            portfolio_values=values,
            asset_order=asset_order,
            safe_eps=safe_eps,
        )

    elif weight_mode == "static":
        if target_weights is None:
            raise ValueError(
                "target_weights must be provided when weight_mode='static'"
            )

        validate_target_weights(target_weights, asset_order)
        initial_portfolio_values = initial_portfolio_values_from_shares(
            prices=prices,
            initial_asset_shares=initial_asset_shares,
            asset_order=asset_order,
        )

        values = portfolio_values_static_weights(
            prices=prices,
            target_weights=target_weights,
            asset_order=asset_order,
            initial_portfolio_values=initial_portfolio_values,
            safe_eps=safe_eps,
        )
        weights = portfolio_weights_forecast_static(
            target_weights=target_weights,
            n_sims=n_sims,
            n_periods=n_periods,
            asset_order=asset_order,
        )

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    pnl = pnl_from_values(
        values=values,
        mode=pnl_type,
    )

    return PortfolioForecast(
        values=values,
        incremental_pnl=pnl,
        pnl_type=pnl_type,
        weight_mode=weight_mode,
        asset_weights=weights,
        path_probs=path_probs,
    )
