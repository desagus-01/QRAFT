import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from construction.optimization.constraints import PortfolioConstraint
from construction.optimization.moments import HorizonMoments, MomentsConfig, PnL_OPTIONS
from construction.optimization.objectives.specs import HoldingCost, TransactionCost
from construction.optimization.optimization import MPOResult
from construction.optimization.presets import PreMadeObjectives
from construction.optimization.problem import MPOProblem
from construction.state import PortfolioState
from forecast.forecast_paths import ForecastPaths
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    asset_order: list[str]
    target_weights_risk: NDArray[np.floating]
    target_cash_weight: float
    cash_return: NDArray[np.floating] | None = None
    diagnostics: Any | None = None

    @property
    def target_weights_risk_dict(self) -> dict[str, NDArray[np.floating]]:
        return dict(zip(self.asset_order, self.target_weights_risk))

    @property
    def total_target_weights(self) -> NDArray[np.floating]:
        return np.append(self.target_weights_risk, self.target_cash_weight)

    @property
    def total_target_weights_dict(self) -> dict[str, NDArray[np.floating]]:
        return dict(zip(self.asset_order + ["cash"], self.total_target_weights))


class PolicyProtocol(Protocol):
    name: str

    def decide(
        self, state: PortfolioState, forecasts: ForecastPaths
    ) -> PolicyDecision: ...


def _decision_from_mpo(
    result: MPOResult, cash_return: NDArray[np.floating]
) -> PolicyDecision:
    return PolicyDecision(
        asset_order=result.assets,
        target_weights_risk=result.target_weights,
        target_cash_weight=result.target_cash,
        cash_return=cash_return,
        diagnostics=result,
    )


@dataclass(frozen=True, slots=True)
class EqualWeightPolicy:
    target_cash_weight: float
    name: str = "equal_weight"

    def decide(self, state: PortfolioState, forecasts: ForecastPaths) -> PolicyDecision:
        risky_weights = 1.0 - self.target_cash_weight
        n_assets = len(state.asset_order)
        target_weights = np.full(n_assets, risky_weights / n_assets)
        return PolicyDecision(
            asset_order=state.asset_order,
            target_weights_risk=target_weights,
            target_cash_weight=self.target_cash_weight,
            cash_return=np.zeros(1),
            diagnostics=None,
        )


@dataclass(frozen=True, slots=True)
class MPOPolicy:
    problem: MPOProblem
    moments_config: MomentsConfig
    name: str = "mpo"

    @classmethod
    def preset(
        cls,
        objective_type: PreMadeObjectives,
        risk_aversion: float,
        cash_path: str,
        *,
        name: str = "mpo",
        horizons: int | None = None,
        pnl_type: PnL_OPTIONS = "relative",
        expectation_tolerance: float | None = 1.0,
        step_size: int = 1,
        periods_per_year: int = 252,
        alpha: float | None = 0.05,
        constraints: Sequence[PortfolioConstraint] = (),
        allow_borrow: bool = False,
        max_iter: int = 200,
        transaction_cost: TransactionCost | None = None,
        transaction_cost_weight: float = 1.0,
        holding_cost: HoldingCost | None = None,
        holding_cost_weight: float = 1.0,
        **solver_options: Any,
    ) -> "MPOPolicy":
        return cls(
            problem=MPOProblem.preset(
                objective_type,
                risk_aversion,
                alpha=alpha,
                constraints=constraints,
                allow_borrow=allow_borrow,
                max_iter=max_iter,
                transaction_cost=transaction_cost,
                transaction_cost_weight=transaction_cost_weight,
                holding_cost=holding_cost,
                holding_cost_weight=holding_cost_weight,
                **solver_options,
            ),
            moments_config=MomentsConfig(
                cash_path=cash_path,
                horizons=horizons,
                pnl_type=pnl_type,
                expectation_tolerance=expectation_tolerance,
                step_size=step_size,
                periods_per_year=periods_per_year,
            ),
            name=name,
        )

    @staticmethod
    def _transfer_dropped_to_cash(
        state: PortfolioState,
        kept_assets: list[str],
    ) -> tuple[float, set[str], float]:
        """Return adjusted cash weight, the set of dropped asset names, and
        the total weight transferred from dropped assets to cash.
        """
        weights_by_asset = state.portfolio_weights_dict
        kept_set = set(kept_assets)
        dropped = {a for a in state.asset_order if a not in kept_set}
        dropped_weight = sum(float(weights_by_asset[a]) for a in dropped)
        adjusted_cash = float(state.cash_weight) + dropped_weight
        return adjusted_cash, dropped, dropped_weight

    def compute_moments(self, forecasts: ForecastPaths) -> HorizonMoments:
        cfg = self.moments_config
        return HorizonMoments.from_forecast_paths(
            forecast_paths=forecasts,
            cash_path=cfg.cash_path,
            horizons=cfg.horizons,
            subset=cfg.subset,
            pnl_type=cfg.pnl_type,
            expectation_tolerance=cfg.expectation_tolerance,
            step_size=cfg.step_size,
            periods_per_year=cfg.periods_per_year,
        )

    def decide(self, state: PortfolioState, forecasts: ForecastPaths) -> PolicyDecision:
        moments = self.compute_moments(forecasts)
        current_cash, dropped, dropped_weight = self._transfer_dropped_to_cash(
            state, moments.assets
        )
        if dropped:
            logger.warning(
                "decide(): %d asset(s) dropped by compute_moments %s — "
                "their combined weight (%.4f) transferred to cash for reallocation.",
                len(dropped),
                sorted(dropped),
                dropped_weight,
            )
        weights_by_asset = state.portfolio_weights_dict
        current_weights = np.array(
            [weights_by_asset[asset] for asset in moments.assets]
        )
        result = self.problem.solve(
            moments=moments,
            current_weights=current_weights,
            current_cash=current_cash,
        )
        return _decision_from_mpo(result, cash_return=moments.cash_return)
