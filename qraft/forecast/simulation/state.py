from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Self

import numpy as np
from numpy._typing import NDArray

from time_series.models.fitted_types import UnivariateRes
from time_series.models.model_types import UnivariateModel


@dataclass(slots=True)
class UnivariateState:
    series_hist: NDArray[np.floating]
    ma_residual_lags: NDArray[np.floating] | None = None
    vol_residual_lags: NDArray[np.floating] | None = None
    var_hist: NDArray[np.floating] | None = None

    @classmethod
    def from_fitting_results_and_model(
        cls,
        fitting_results: UnivariateRes,
        univariate_model: UnivariateModel,
        post_series_non_null: NDArray[np.floating],
        x_hist_len: int = 10,
    ) -> Self:
        """
        Build an initial UnivariateState from fitting results and observed series.

        The returned state contains:
          - series_hist: recent history of the series (used to seed mean models)
          - ma_residual_lags: recent MA residuals if a mean model was fitted
          - vol_residual_lags: recent volatility residuals if a GARCH model was fitted
          - var_hist: recent conditional variance history for GARCH variance lags

        Parameters
        ----------
        fitting_results : UnivariateRes
            Fitted univariate results containing mean and volatility fit objects.
        univariate_model : UnivariateModel
            Lightweight description of the univariate model (orders and kinds).
        post_series_non_null : NDArray[np.floating]
            Non-null post-processed series used to extract recent history.
        x_hist_len : int, optional
            Number of recent observations to keep for seeding simulations (default: 10).

        Returns
        -------
        UnivariateState
            Initialized state object with arrays prepared for simulation.
        """
        x_hist = post_series_non_null[
            -min(x_hist_len, post_series_non_null.size) :
        ].copy()

        eps_mean_hist = None
        if (
            univariate_model.mean_kind == "arma"
            and fitting_results.mean_res is not None
        ):
            p, q = univariate_model.mean_order
            if x_hist.size < p:
                raise ValueError("Full series is too short, longer input is needed.")
            eps = fitting_results.mean_res.residuals
            eps_mean_hist = eps[-q:].copy() if q > 0 else None

        eps_vol_hist = None
        var_hist = None
        if (
            univariate_model.vol_kind == "garch"
            and fitting_results.volatility_res is not None
        ):
            p_g, o_g, q_g = univariate_model.vol_order
            m = max(p_g, o_g, 1)
            sig2 = fitting_results.volatility_res.conditional_volatility**2
            eps_vol_hist = (
                fitting_results.volatility_res.residuals[-m:].copy() if m > 0 else None
            )
            var_hist = sig2[-q_g:].copy() if q_g > 0 else None

        return cls(
            series_hist=x_hist,
            ma_residual_lags=eps_mean_hist,
            vol_residual_lags=eps_vol_hist,
            var_hist=var_hist,
        )

    def state_as_dict(self) -> Mapping[str, NDArray[np.floating]]:
        """
        Return a dict representation of the state suitable for inspection/logging.

        Returns
        -------
        Mapping[str, NDArray[np.floating]]
            Dictionary with keys 'series_hist', 'ma_residual_lags',
            'vol_residual_lags' and 'var_hist' mapping to numpy arrays or None.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SimulationForecast:
    model: UnivariateModel
    state0: UnivariateState

    @classmethod
    def from_res_and_series(
        cls,
        fitting_results: UnivariateRes,
        post_series_non_null: NDArray[np.floating],
        x_hist_len: int = 10,
    ):
        """
        Construct a SimulationForecast from fitting results and observed series.

        This helper creates a :class:`UnivariateModel` from the fitting results
        and prepares the initial state used by simulation routines. If neither
        a mean nor volatility model is available in ``fitting_results``, a
        fallback ``innovation_scale`` is computed from first differences of
        the input series to allow simulation of random-walk like behaviour.

        Parameters
        ----------
        fitting_results : UnivariateRes
            Fitted mean/volatility results for the asset.
        post_series_non_null : NDArray[np.floating]
            Array of non-null post-processed observations for the asset.
        x_hist_len : int, optional
            Length of series history to keep for state initialization (default: 10).

        Returns
        -------
        SimulationForecast
            Frozen dataclass containing the univariate model and initial state.

        Raises
        ------
        ValueError
            If ``post_series_non_null`` is empty.
        """
        if post_series_non_null.size == 0:
            raise ValueError("post_series_non_null is empty")

        model = UnivariateModel.from_fitting_results(fitting_results=fitting_results)

        if fitting_results.mean_res is None and fitting_results.volatility_res is None:
            diff = np.diff(post_series_non_null)
            scale = float(np.std(diff, ddof=1)) if diff.size > 0 else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0

            model = UnivariateModel(
                mean_kind=model.mean_kind,
                mean_order=model.mean_order,
                mean_params=model.mean_params,
                vol_kind=model.vol_kind,
                vol_order=model.vol_order,
                vol_params=model.vol_params,
                innovation_scale=scale,
            )

        return cls(
            model=model,
            state0=UnivariateState.from_fitting_results_and_model(
                fitting_results=fitting_results,
                univariate_model=model,
                post_series_non_null=post_series_non_null,
                x_hist_len=x_hist_len,
            ),
        )
