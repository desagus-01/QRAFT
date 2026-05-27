import logging
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

import cvxpy as cp
import numpy as np
from cvxpy import Constraint, Expression
from numpy.typing import NDArray
from portfolio.policy import PortfolioConstraint
from portfolio.policy.moments import HorizonMoments
from portfolio.policy.objectives.protocol import (
    get_objective_handler,
    get_refineable_handler,
)
from portfolio.policy.objectives.specs import (
    CovarianceRisk,
    CVaRCuttingPlane,
    CVaRRisk,
    ExpectedReturn,
    HoldingCost,
    ObjectiveSpec,
    TransactionCost,
    WeightedTerm,
)

logger = logging.getLogger(__name__)

PreMadeObjectives = Literal["mean_covariance", "cvar_auto", "cvar_classic", "cvar_cuts"]


SolverStatus = Literal[
    "optimal",
    "optimal_inaccurate",
    "infeasible",
    "infeasible_inaccurate",
    "unbounded",
    "unbounded_inaccurate",
    "solver_error",
]


def mean_covariance_objectives(risk_aversion: float) -> ObjectiveSpec:
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn(decay=0.9)),
            WeightedTerm(risk_aversion, CovarianceRisk()),
            WeightedTerm(
                1.0,
                TransactionCost(
                    cost=0.0005,
                    pershare_cost=0.005,
                    market_impact=0.8,
                    exponent=1.5,
                    c_bias=0.0003,
                ),
            ),
            WeightedTerm(
                1.0,
                HoldingCost(
                    short_fees=0.0,
                    long_fees=0.3,
                    dividends=0.0,
                    periods_per_year=252,
                ),
            ),
        )
    )


def cvar_classical_objectives(
    cvar_aversion: float, alpha: float = 0.05
) -> ObjectiveSpec:
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(cvar_aversion, CVaRRisk(alpha=alpha)),
            WeightedTerm(
                1.0,
                TransactionCost(
                    cost=0.0005,
                    pershare_cost=0.005,
                    market_impact=0.8,
                    exponent=1.5,
                    c_bias=0.0003,
                ),
            ),
            WeightedTerm(
                1.0,
                HoldingCost(
                    short_fees=0.0,
                    long_fees=0.3,
                    dividends=0.0,
                    periods_per_year=252,
                ),
            ),
        )
    )


def cvar_cuts_objectives(cvar_aversion: float, alpha: float = 0.05) -> ObjectiveSpec:
    return ObjectiveSpec(
        terms=(
            WeightedTerm(1.0, ExpectedReturn()),
            WeightedTerm(cvar_aversion, CVaRCuttingPlane(alpha=alpha)),
            WeightedTerm(
                1.0,
                TransactionCost(
                    cost=0.0005,
                    pershare_cost=0.005,
                    market_impact=0.8,
                    exponent=1.5,
                    c_bias=0.0003,
                ),
            ),
            WeightedTerm(
                1.0,
                HoldingCost(
                    short_fees=0.0,
                    long_fees=0.3,
                    dividends=0.0,
                    periods_per_year=252,
                ),
            ),
        )
    )


def _select_cvar_solver(
    horizons: int, n_scenarios: int, problem_limit: int = 1_000
) -> PreMadeObjectives:
    """Auto-pick CVaR formulation from problem size."""
    problem_scale = horizons * n_scenarios
    return "cvar_cuts" if problem_scale >= problem_limit else "cvar_classic"


def _map_type_to_objective(
    objective_type: PreMadeObjectives,
    risk_aversion: float,
    cvar_alpha: float | None,
) -> ObjectiveSpec:
    if objective_type == "mean_covariance":
        return mean_covariance_objectives(risk_aversion=risk_aversion)
    if objective_type == "cvar_classic" and cvar_alpha is not None:
        return cvar_classical_objectives(cvar_aversion=risk_aversion, alpha=cvar_alpha)
    if objective_type == "cvar_cuts" and cvar_alpha is not None:
        return cvar_cuts_objectives(cvar_aversion=risk_aversion, alpha=cvar_alpha)
    raise ValueError(
        f"Your {objective_type} is not valid, please choose a suitable one."
    )


def _structural_constraints(
    horizon_trades: Expression,
    horizon_weights: Expression,
    previous_weights: NDArray[np.floating] | Expression,
) -> list[Constraint]:
    """Self-financing and horizon-linking constraints."""
    return cast(
        list[Constraint],
        [
            cp.sum(horizon_trades) == 0,
            horizon_weights == previous_weights + horizon_trades,
        ],
    )


# TODO: TMI, make this smaller and more useful
@dataclass(frozen=True, slots=True)
class MPOResult:
    """
    Outcome of a multi-period portfolio optimization.

    The optimizer plans an entire (n_horizons, n_assets) weight path, but only
    row 0 is actionable. The rest is informational.
    """

    assets: list[str]
    planned_weights: NDArray[np.floating]
    planned_trades: NDArray[np.floating]
    initial_weights: NDArray[np.floating]
    status: SolverStatus
    objective_value: float
    solver_stats: Any  # cvxpy.SolverStats; loose to avoid hard coupling

    def __post_init__(self) -> None:
        n_assets = len(self.assets)
        if self.planned_weights.ndim != 2 or self.planned_weights.shape[1] != n_assets:
            raise ValueError(
                f"planned_weights must have shape (n_horizons, {n_assets}); "
                f"got {self.planned_weights.shape}"
            )
        if self.planned_trades.shape != self.planned_weights.shape:
            raise ValueError(
                f"planned_trades shape {self.planned_trades.shape} must match "
                f"planned_weights shape {self.planned_weights.shape}"
            )
        if self.initial_weights.shape != (n_assets,):
            raise ValueError(
                f"initial_weights must have shape ({n_assets},); "
                f"got {self.initial_weights.shape}"
            )

    @property
    def n_horizons(self) -> int:
        return self.planned_weights.shape[0]

    @property
    def is_optimal(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate")

    @property
    def target_weights(self) -> NDArray[np.floating]:
        """Post-trade weights to rebalance to now."""
        return self.planned_weights[0]

    @property
    def target_weights_by_asset(self) -> dict[str, float]:
        """
        ``target_weights`` keyed by asset name.

        This is the dict shape consumed by
        ``portfolio_forecast(weight_mode="static", target_weights=...)``.
        """
        return dict(zip(self.assets, self.target_weights.tolist()))

    @property
    def first_trade(self) -> NDArray[np.floating]:
        return self.planned_trades[0]

    @property
    def first_trade_by_asset(self) -> dict[str, float]:
        return dict(zip(self.assets, self.first_trade.tolist()))

    @property
    def turnover(self) -> float:
        """One-way turnover of the first trade: 0.5 * ||first_trade||_1."""
        return 0.5 * float(np.abs(self.first_trade).sum())

    def weights_at_horizon(self, horizon: int) -> NDArray[np.floating]:
        self._check_horizon(horizon)
        return self.planned_weights[horizon]

    def trades_at_horizon(self, horizon: int) -> NDArray[np.floating]:
        self._check_horizon(horizon)
        return self.planned_trades[horizon]

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
    ) -> None:
        self.objective = objective
        self.horizons = horizons
        self.n_assets = n_assets
        self.n_scenarios = n_scenarios
        self.constraints = constraints
        self.weights = cp.Variable((horizons, n_assets), name="weights")
        self.trades = cp.Variable((horizons, n_assets), name="trades")
        self.current_weights = cp.Parameter(n_assets, name="current_weights")
        self._term_params: list[dict[str, Any]] = []
        for weighted_term in objective.terms:
            term_handler = get_objective_handler(weighted_term.spec)
            self._term_params.append(
                term_handler.allocate(
                    weighted_term.spec, horizons, n_assets, n_scenarios=n_scenarios
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
    ) -> "MultiPeriodOptimizer":
        """
        Compile and return a :class:`MultiPeriodOptimizer` from a named
        pre-built objective recipe — **without solving it**.

        Parameters
        ----------
        objective_type:
            Named objective recipe.  ``"cvar_auto"`` is resolved from
            ``horizons * n_scenarios`` before the problem is built.
        horizons:
            Number of look-ahead periods.
        n_assets:
            Number of investable assets.
        n_scenarios:
            Number of scenario paths used in CVaR objectives.
        risk_aversion:
            Scalar risk-aversion / CVaR-aversion coefficient.
        cvar_alpha:
            Tail probability for CVaR objectives.  Unused for
            ``"mean_covariance"``.
        constraints:
            Optional hard / soft portfolio constraints.

        Returns
        -------
        MultiPeriodOptimizer
            Fully compiled, ready to call ``.solve()`` /
            ``.solve_iterative()`` many times.
        """
        resolved_objective_type: PreMadeObjectives = (
            _select_cvar_solver(horizons=horizons, n_scenarios=n_scenarios)
            if objective_type == "cvar_auto"
            else objective_type
        )
        objective = _map_type_to_objective(
            objective_type=resolved_objective_type,
            risk_aversion=risk_aversion,
            cvar_alpha=cvar_alpha,
        )
        return cls(
            objective=objective,
            horizons=horizons,
            n_assets=n_assets,
            constraints=constraints,
            n_scenarios=n_scenarios,
        )

    def _build_problem(self) -> None:
        previous_weights = self.current_weights
        objective_expressions: list[cp.Expression] = []
        compiled_constraints: list[cp.Constraint] = []

        for horizon_idx in range(self.horizons):
            horizon_weights, horizon_trades = (
                self.weights[horizon_idx, :],
                self.trades[horizon_idx, :],
            )
            compiled_constraints += _structural_constraints(
                horizon_trades, horizon_weights, previous_weights
            )
            if self.constraints is not None:
                for constraint in self.constraints:
                    if constraint.constraint_type == "hard":
                        compiled_constraints += constraint.compile_to_cvxpy(
                            horizon_weights, horizon_trades
                        )
                    else:  # ie soft
                        constraint_violation = constraint.violation_expr(
                            horizon_weights, horizon_trades
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
                )
                objective_expressions.append(weighted_term.weight * term_expression)
                compiled_constraints += aux_constraints

            previous_weights = horizon_weights

        self.problem = cp.Problem(
            cp.Maximize(cp.sum(objective_expressions)), compiled_constraints
        )

    def solve(
        self,
        moments: HorizonMoments,
        current_weights: NDArray[np.floating],
        inputs: dict[str, Any] | None = None,
        **solver_options,
    ) -> MPOResult:
        self.current_weights.value = current_weights
        solver_inputs = {"moments": moments, **(inputs or {})}

        for weighted_term, term_params in zip(self.objective.terms, self._term_params):
            get_objective_handler(weighted_term.spec).update(
                weighted_term.spec, term_params, solver_inputs
            )

        self.problem.solve(
            enforce_dpp=True,
            warm_start=True,
            **solver_options,
        )
        if self.problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Optimization failed: {self.problem.status}")

        optimal_weights = self.weights.value
        optimal_trades = self.trades.value
        optimal_objective_value = self.problem.value

        if (
            optimal_weights is None
            or optimal_trades is None
            or optimal_objective_value is None
        ):
            raise RuntimeError(
                "Solver returned None values — unexpected state after optimal solve."
            )

        return MPOResult(
            assets=moments.assets,
            planned_weights=optimal_weights,
            planned_trades=optimal_trades,
            initial_weights=current_weights,
            status=cast(SolverStatus, self.problem.status),
            objective_value=float(optimal_objective_value),
            solver_stats=self.problem.solver_stats,
        )

    def solve_iterative(
        self,
        moments: HorizonMoments,
        current_weights: NDArray[np.floating],
        inputs: dict[str, Any] | None = None,
        max_iter: int = 200,
        **solver_options,
    ) -> MPOResult:
        self.current_weights.value = current_weights
        solver_inputs = {"moments": moments, **(inputs or {})}

        # Initial parameter setup (all handlers)
        for weighted_term, term_params in zip(self.objective.terms, self._term_params):
            get_objective_handler(weighted_term.spec).update(
                weighted_term.spec, term_params, solver_inputs
            )

        # Identify cutting-plane CVaR terms that need iterative refinement
        cutting_plane_terms = [
            (weighted_term, term_params)
            for weighted_term, term_params in zip(
                self.objective.terms, self._term_params
            )
            if isinstance(weighted_term.spec, CVaRCuttingPlane)
        ]

        optimal_weights: NDArray[np.floating] | None = None
        optimal_trades: NDArray[np.floating] | None = None
        optimal_objective_value: float | None = None

        for iteration in range(max_iter):
            self.problem.solve(enforce_dpp=True, warm_start=True, **solver_options)

            if self.problem.status not in {"optimal", "optimal_inaccurate"}:
                raise RuntimeError(
                    f"Iteration {iteration}: solve failed with {self.problem.status}"
                )

            optimal_weights = self.weights.value
            optimal_trades = self.trades.value
            optimal_objective_value = self.problem.value

            if not cutting_plane_terms:
                break  # no cutting-plane CVaR → one-shot

            converged = True
            for weighted_term, term_params in cutting_plane_terms:
                term_handler = get_refineable_handler(weighted_term.spec)
                assert optimal_weights is not None
                if not term_handler.refine(
                    weighted_term.spec, term_params, optimal_weights, moments
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
        if (
            optimal_weights is None
            or optimal_trades is None
            or optimal_objective_value is None
        ):
            raise RuntimeError(
                "Solver returned None values — unexpected state after optimal solve."
            )

        return MPOResult(
            assets=moments.assets,
            planned_weights=optimal_weights,
            planned_trades=optimal_trades,
            initial_weights=current_weights,
            status=cast(SolverStatus, self.problem.status),
            objective_value=float(optimal_objective_value),
            solver_stats=self.problem.solver_stats,
        )
