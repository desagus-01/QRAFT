from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

import cvxpy as cp
import numpy as np
from construction.optimization.constraints import PortfolioConstraint
from construction.optimization.moments import HorizonMoments
from construction.optimization.objectives import (
    handlers as _objective_handlers,  # noqa: F401
)
from construction.optimization.objectives.protocol import (
    get_objective_handler,
    get_refineable_handler,
)
from construction.optimization.objectives.specs import (
    CVaRCuttingPlane,
    ObjectiveSpec,
)
from construction.optimization.presets import (
    PreMadeObjectives,
    _build_preset_objective,
)
from cvxpy import Constraint, Expression
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


SolverStatus = Literal[
    "optimal",
    "optimal_inaccurate",
    "infeasible",
    "infeasible_inaccurate",
    "unbounded",
    "unbounded_inaccurate",
    "solver_error",
]


def _structural_constraints(
    horizon_trades: Expression,
    horizon_weights: Expression,
    previous_weights: NDArray[np.floating] | Expression,
    horizon_cash_trade: Expression,
    horizon_cash: Expression,
    previous_cash: float | Expression,
) -> list[Constraint]:
    """
    Self-financing and horizon-linking constraints with explicit scalar cash.
    """
    return cast(
        list[Constraint],
        [
            cp.sum(horizon_trades) + horizon_cash_trade == 0,
            horizon_weights == previous_weights + horizon_trades,
            horizon_cash == previous_cash + horizon_cash_trade,
        ],
    )


def _validate_current_state(
    current_weights: NDArray[np.floating],
    current_cash: float,
    n_assets: int,
    allow_borrow: bool,
) -> tuple[NDArray[np.floating], float]:
    if current_weights.shape != (n_assets,):
        raise ValueError(
            f"current_weights must have shape ({n_assets},); got {current_weights.shape}."
        )

    if not np.all(np.isfinite(current_weights)):
        raise ValueError("current_weights contains NaN or inf.")

    total_budget = float(np.sum(current_weights)) + current_cash

    if not np.isclose(total_budget, 1.0, atol=1e-6):
        raise ValueError(
            "Initial risky weights plus current_cash must sum to 1. "
            f"Got {total_budget:.12f}."
        )

    if not allow_borrow and current_cash < -1e-10:
        raise ValueError(
            "current_cash is negative but allow_borrow=False. "
            f"Got current_cash={current_cash:.12f}."
        )

    return current_weights, current_cash


@dataclass(frozen=True, slots=True)
class MPOResult:
    """
    Outcome of a multi-period portfolio optimization.

    The optimizer plans an entire risky-weight path plus an explicit cash path.
    Only horizon 0 is actionable; later rows are informational.
    """

    assets: list[str]
    planned_weights: NDArray[np.floating]
    planned_trades: NDArray[np.floating]
    planned_cash: NDArray[np.floating]
    planned_cash_trades: NDArray[np.floating]
    initial_weights: NDArray[np.floating]
    initial_cash: float
    status: SolverStatus
    objective_value: float
    solver_stats: Any

    def __post_init__(self) -> None:
        n_assets = len(self.assets)

        if self.planned_weights.ndim != 2 or self.planned_weights.shape[1] != n_assets:
            raise ValueError(
                f"planned_weights must have shape (n_horizons, {n_assets}); "
                f"got {self.planned_weights.shape}."
            )

        if self.planned_trades.shape != self.planned_weights.shape:
            raise ValueError(
                f"planned_trades shape {self.planned_trades.shape} must match "
                f"planned_weights shape {self.planned_weights.shape}."
            )

        if self.initial_weights.shape != (n_assets,):
            raise ValueError(
                f"initial_weights must have shape ({n_assets},); "
                f"got {self.initial_weights.shape}."
            )

        expected_cash_shape = (self.planned_weights.shape[0],)

        if self.planned_cash.shape != expected_cash_shape:
            raise ValueError(
                f"planned_cash must have shape {expected_cash_shape}; "
                f"got {self.planned_cash.shape}."
            )

        if self.planned_cash_trades.shape != expected_cash_shape:
            raise ValueError(
                f"planned_cash_trades must have shape {expected_cash_shape}; "
                f"got {self.planned_cash_trades.shape}."
            )

        if not np.isfinite(self.initial_cash):
            raise ValueError("initial_cash must be finite.")

    @property
    def n_horizons(self) -> int:
        return self.planned_weights.shape[0]

    @property
    def is_optimal(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate")

    @property
    def target_weights(self) -> NDArray[np.floating]:
        """Post-trade risky weights to rebalance to now."""
        return self.planned_weights[0]

    @property
    def target_cash(self) -> float:
        """Post-trade cash weight to hold now."""
        return float(self.planned_cash[0])

    @property
    def target_weights_by_asset(self) -> dict[str, float]:
        """
        Risky-only target weights keyed by asset.

        Note: with explicit cash, this dictionary may sum to less than 1.
        """
        return dict(zip(self.assets, self.target_weights.tolist()))

    def target_weights_renormalized(
        self, small_holding_limit: float = 0.01
    ) -> dict[str, float]:
        """
        Risky-only weights, where very small holdings are dropped and renormalized to sum to 1.
        """
        risky_sum = float(np.sum(self.target_weights))
        normalized = self.target_weights / risky_sum
        filtered = {
            asset: w
            for asset, w in zip(self.assets, normalized.tolist())
            if w > small_holding_limit
        }

        total = sum(filtered.values())
        return {asset: w / total for asset, w in filtered.items()}

    def target_weights_by_asset_with_cash(
        self,
        cash_key: str = "CASH",
    ) -> dict[str, float]:
        """
        Target weights keyed by asset, including explicit cash.

        The resulting dictionary should sum to 1 up to solver tolerance.
        """
        out = self.target_weights_by_asset
        out[cash_key] = self.target_cash
        return out

    @property
    def first_trade(self) -> NDArray[np.floating]:
        return self.planned_trades[0]

    @property
    def first_cash_trade(self) -> float:
        return float(self.planned_cash_trades[0])

    @property
    def first_trade_by_asset(self) -> dict[str, float]:
        return dict(zip(self.assets, self.first_trade.tolist()))

    @property
    def turnover(self) -> float:
        """
        Cash-aware one-way turnover of the first trade:

            0.5 * (||risky_trade||_1 + |cash_trade|).
        """
        return 0.5 * float(np.abs(self.first_trade).sum() + abs(self.first_cash_trade))

    def weights_at_horizon(self, horizon: int) -> NDArray[np.floating]:
        self._check_horizon(horizon)
        return self.planned_weights[horizon]

    def trades_at_horizon(self, horizon: int) -> NDArray[np.floating]:
        self._check_horizon(horizon)
        return self.planned_trades[horizon]

    def cash_at_horizon(self, horizon: int) -> float:
        self._check_horizon(horizon)
        return float(self.planned_cash[horizon])

    def cash_trade_at_horizon(self, horizon: int) -> float:
        self._check_horizon(horizon)
        return float(self.planned_cash_trades[horizon])

    def _check_horizon(self, horizon: int) -> None:
        if not 0 <= horizon < self.n_horizons:
            raise ValueError(f"horizon must be in 0..{self.n_horizons - 1}")


class MultiPeriodOptimizer:
    def __init__(
        self,
        objective: ObjectiveSpec,
        horizons: int,
        n_assets: int,
        n_scenarios: int,
        constraints: Sequence[PortfolioConstraint] | None = None,
        allow_borrow: bool = False,
    ) -> None:
        self.objective = objective
        self.horizons = horizons
        self.n_assets = n_assets
        self.n_scenarios = n_scenarios
        self.constraints = constraints
        self.allow_borrow = allow_borrow

        # Risky-only variables — shape (horizons, n_assets).
        self.weights = cp.Variable((horizons, n_assets), name="weights")
        self.trades = cp.Variable((horizons, n_assets), name="trades")
        self.current_weights = cp.Parameter(n_assets, name="current_weights")

        # Explicit scalar cash variables — shape (horizons,).
        self.cash = cp.Variable(horizons, name="cash")
        self.cash_trade = cp.Variable(horizons, name="cash_trade")
        self.current_cash = cp.Parameter(name="current_cash")

        self._term_params: list[dict[str, Any]] = []
        for weighted_term in objective.terms:
            term_handler = get_objective_handler(weighted_term.spec)
            self._term_params.append(
                term_handler.allocate(
                    weighted_term.spec,
                    horizons,
                    n_assets,
                    n_scenarios=n_scenarios,
                )
            )

        self._build_problem()

    @classmethod
    def from_pre_built(
        cls,
        objective_type: PreMadeObjectives,
        horizons: int,
        n_assets: int,
        n_scenarios: int,
        risk_aversion: float,
        cvar_alpha: float | None = 0.05,
        constraints: Sequence[PortfolioConstraint] | None = None,
        allow_borrow: bool = False,
    ) -> "MultiPeriodOptimizer":
        """
        Compile and return a :class:`MultiPeriodOptimizer` from a named
        pre-built objective recipe — **without solving it**.
        """
        objective = _build_preset_objective(
            objective_type=objective_type,
            risk_aversion=risk_aversion,
            alpha=cvar_alpha,
            horizons=horizons,
            n_scenarios=n_scenarios,
        )
        return cls(
            objective=objective,
            horizons=horizons,
            n_assets=n_assets,
            constraints=constraints,
            n_scenarios=n_scenarios,
            allow_borrow=allow_borrow,
        )

    @property
    def uses_cutting_plane(self) -> bool:
        return any(
            isinstance(weighted_term.spec, CVaRCuttingPlane)
            for weighted_term in self.objective.terms
        )

    def _build_problem(self) -> None:
        previous_weights: NDArray[np.floating] | Expression = self.current_weights
        previous_cash: float | Expression = self.current_cash

        objective_expressions: list[cp.Expression] = []
        compiled_constraints: list[cp.Constraint] = []

        for horizon_idx in range(self.horizons):
            horizon_weights = self.weights[horizon_idx, :]
            horizon_trades = self.trades[horizon_idx, :]
            horizon_cash = self.cash[horizon_idx]
            horizon_cash_trade = self.cash_trade[horizon_idx]

            compiled_constraints += _structural_constraints(
                horizon_trades=horizon_trades,
                horizon_weights=horizon_weights,
                previous_weights=previous_weights,
                horizon_cash_trade=horizon_cash_trade,
                horizon_cash=horizon_cash,
                previous_cash=previous_cash,
            )

            if not self.allow_borrow:
                compiled_constraints.append(horizon_cash >= 0)

            if self.constraints is not None:
                for constraint in self.constraints:
                    if constraint.constraint_type == "hard":
                        compiled_constraints += constraint.compile_to_cvxpy(
                            horizon_weights,
                            horizon_trades,
                            cash_trade_h=horizon_cash_trade,
                        )
                    else:
                        constraint_violation = constraint.violation_expr(
                            horizon_weights,
                            horizon_trades,
                            cash_trade_h=horizon_cash_trade,
                        )
                        objective_expressions.append(
                            -constraint.soft_weight * cp.sum(constraint_violation)
                        )

            for weighted_term, term_params in zip(
                self.objective.terms, self._term_params
            ):
                term_handler = get_objective_handler(weighted_term.spec)
                term_expression, aux_constraints = term_handler.compile(
                    weighted_term.spec,
                    term_params,
                    horizon_weights,
                    horizon_trades,
                    horizon_idx,
                    cash_h=horizon_cash,
                )
                objective_expressions.append(weighted_term.weight * term_expression)
                compiled_constraints += aux_constraints

            previous_weights = horizon_weights
            previous_cash = horizon_cash

        self.problem = cp.Problem(
            cp.Maximize(cp.sum(objective_expressions)), compiled_constraints
        )

    def _set_current_state(
        self,
        current_weights: NDArray[np.floating],
        current_cash: float | None,
    ) -> tuple[NDArray[np.floating], float]:
        """
        Validate and load current portfolio state into CVXPY parameters.

        If ``current_cash`` is ``None``, it is inferred as
        ``1 - sum(current_weights)``.
        """
        if current_cash is None:
            current_cash = 1.0 - float(np.sum(current_weights))

        weights, cash = _validate_current_state(
            current_weights=current_weights,
            current_cash=current_cash,
            n_assets=self.n_assets,
            allow_borrow=self.allow_borrow,
        )

        self.current_weights.value = weights
        self.current_cash.value = cash

        return weights, cash

    def _update_parameters(
        self,
        moments: HorizonMoments,
        inputs: dict[str, Any] | None,
    ) -> None:
        solver_inputs = {"moments": moments, **(inputs or {})}

        for weighted_term, term_params in zip(self.objective.terms, self._term_params):
            get_objective_handler(weighted_term.spec).update(
                weighted_term.spec,
                term_params,
                solver_inputs,
            )

    def _build_result(
        self,
        moments: HorizonMoments,
        initial_weights: NDArray[np.floating],
        initial_cash: float,
        objective_value: float,
    ) -> MPOResult:
        optimal_weights = self.weights.value
        optimal_trades = self.trades.value
        optimal_cash = self.cash.value
        optimal_cash_trades = self.cash_trade.value

        if (
            optimal_weights is None
            or optimal_trades is None
            or optimal_cash is None
            or optimal_cash_trades is None
        ):
            raise RuntimeError(
                "Solver returned None values — unexpected state after optimal solve."
            )

        return MPOResult(
            assets=moments.assets,
            planned_weights=optimal_weights,
            planned_trades=optimal_trades,
            planned_cash=optimal_cash,
            planned_cash_trades=optimal_cash_trades,
            initial_weights=initial_weights,
            initial_cash=initial_cash,
            status=cast(SolverStatus, self.problem.status),
            objective_value=objective_value,
            solver_stats=self.problem.solver_stats,
        )

    def solve(
        self,
        moments: HorizonMoments,
        current_weights: NDArray[np.floating],
        current_cash: float | None = None,
        inputs: dict[str, Any] | None = None,
        **solver_options,
    ) -> MPOResult:
        initial_weights, initial_cash = self._set_current_state(
            current_weights=current_weights,
            current_cash=current_cash,
        )

        self._update_parameters(moments=moments, inputs=inputs)

        self.problem.solve(
            enforce_dpp=True,
            warm_start=True,
            **solver_options,
        )

        if self.problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Optimization failed: {self.problem.status}")

        objective_value = self.problem.value

        if objective_value is None:
            raise RuntimeError(
                "Solver returned None objective value — unexpected state after "
                "optimal solve."
            )

        result = self._build_result(
            moments=moments,
            initial_weights=initial_weights,
            initial_cash=initial_cash,
            objective_value=float(objective_value),
        )
        logger.info(
            "MPO solve complete: status=%s objective=%.6f turnover=%.4f "
            "n_assets=%d n_horizons=%d",
            result.status,
            result.objective_value,
            result.turnover,
            len(result.assets),
            result.n_horizons,
        )
        return result

    def solve_iterative(
        self,
        moments: HorizonMoments,
        current_weights: NDArray[np.floating],
        current_cash: float | None = None,
        inputs: dict[str, Any] | None = None,
        max_iter: int = 200,
        **solver_options,
    ) -> MPOResult:
        initial_weights, initial_cash = self._set_current_state(
            current_weights=current_weights,
            current_cash=current_cash,
        )

        self._update_parameters(moments=moments, inputs=inputs)

        cutting_plane_terms = [
            (weighted_term, term_params)
            for weighted_term, term_params in zip(
                self.objective.terms, self._term_params
            )
            if isinstance(weighted_term.spec, CVaRCuttingPlane)
        ]

        objective_value: float | None = None

        for iteration in range(max_iter):
            self.problem.solve(enforce_dpp=True, warm_start=True, **solver_options)

            if self.problem.status not in {"optimal", "optimal_inaccurate"}:
                raise RuntimeError(
                    f"Iteration {iteration}: solve failed with {self.problem.status}"
                )

            objective_value = self.problem.value

            if objective_value is None:
                raise RuntimeError(
                    "Solver returned None objective value — unexpected state after "
                    "optimal solve."
                )

            if not cutting_plane_terms:
                break

            optimal_weights = self.weights.value

            if optimal_weights is None:
                raise RuntimeError(
                    "Solver returned None weights — unexpected state after "
                    "optimal solve."
                )

            converged = True

            for weighted_term, term_params in cutting_plane_terms:
                term_handler = get_refineable_handler(weighted_term.spec)
                if not term_handler.refine(
                    weighted_term.spec,
                    term_params,
                    optimal_weights,
                    moments,
                ):
                    converged = False

            if converged:
                break
        else:
            total_cuts = sum(
                sum(term_params.get("cut_count", [0]))
                for _, term_params in cutting_plane_terms
            )
            logger.warning(
                "CVaR cutting-plane did not converge in %d iterations "
                "(%d total cuts placed).",
                max_iter,
                total_cuts,
            )

        if objective_value is None:
            raise RuntimeError(
                "Solver returned None objective value — unexpected state after "
                "optimal solve."
            )

        result = self._build_result(
            moments=moments,
            initial_weights=initial_weights,
            initial_cash=initial_cash,
            objective_value=float(objective_value),
        )
        logger.info(
            "MPO cutting-plane solve complete: status=%s objective=%.6f "
            "iterations=%d turnover=%.4f n_assets=%d n_horizons=%d",
            result.status,
            result.objective_value,
            iteration + 1,
            result.turnover,
            len(result.assets),
            result.n_horizons,
        )
        return result
