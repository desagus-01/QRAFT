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
    type: PreMadeObjectives,
    risk_aversion: float,
    cvar_alpha: float | None,
) -> ObjectiveSpec:
    if type == "mean_covariance":
        return mean_covariance_objectives(risk_aversion=risk_aversion)
    if type == "cvar_classic" and cvar_alpha is not None:
        return cvar_classical_objectives(cvar_aversion=risk_aversion, alpha=cvar_alpha)
    if type == "cvar_cuts" and cvar_alpha is not None:
        return cvar_cuts_objectives(cvar_aversion=risk_aversion, alpha=cvar_alpha)
    raise ValueError(f"Your {type} is not valid, please choose a suitable one.")


def _structural_constraints(
    trades: Expression,
    weights: Expression,
    previous_weights: NDArray[np.floating] | Expression,
) -> list[Constraint]:
    """Self-financing and horizon-linking constraints."""
    return cast(
        list[Constraint],
        [
            cp.sum(trades) == 0,
            weights == previous_weights + trades,
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
        for term in objective.terms:
            handler = get_objective_handler(term.spec)
            self._term_params.append(
                handler.allocate(term.spec, horizons, n_assets, n_scenarios=n_scenarios)
            )

        self._build_problem()

    @classmethod
    def from_pre_built(
        cls,
        type: PreMadeObjectives,
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
        type:
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
        resolved_type: PreMadeObjectives = (
            _select_cvar_solver(horizons=horizons, n_scenarios=n_scenarios)
            if type == "cvar_auto"
            else type
        )
        objective = _map_type_to_objective(
            type=resolved_type,
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
        prev = self.current_weights
        terms: list[cp.Expression] = []
        constraints: list[cp.Constraint] = []

        for h in range(self.horizons):
            w_h, z_h = self.weights[h, :], self.trades[h, :]
            constraints += _structural_constraints(z_h, w_h, prev)
            if self.constraints is not None:
                for c in self.constraints:
                    if c.constraint_type == "hard":
                        constraints += c.compile_to_cvxpy(w_h, z_h)
                    else:  # ie soft
                        violation = c.violation_expr(w_h, z_h)
                        terms.append(-c.soft_weight * cp.sum(violation))

            for term, params in zip(self.objective.terms, self._term_params):
                handler = get_objective_handler(term.spec)
                expr, aux_constraints = handler.compile(term.spec, params, w_h, z_h, h)
                terms.append(term.weight * expr)
                constraints += aux_constraints

            prev = w_h

        self.problem = cp.Problem(cp.Maximize(cp.sum(terms)), constraints)

    def solve(
        self,
        moments: HorizonMoments,
        current_weights: NDArray[np.floating],
        inputs: dict[str, Any] | None = None,
        **solver_options,
    ) -> MPOResult:
        self.current_weights.value = current_weights
        full_inputs = {"moments": moments, **(inputs or {})}

        for term, params in zip(self.objective.terms, self._term_params):
            get_objective_handler(term.spec).update(term.spec, params, full_inputs)

        self.problem.solve(
            enforce_dpp=True,
            warm_start=True,
            **solver_options,
        )
        if self.problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Optimization failed: {self.problem.status}")

        weights_val = self.weights.value
        trades_val = self.trades.value
        obj_val = self.problem.value

        if weights_val is None or trades_val is None or obj_val is None:
            raise RuntimeError(
                "Solver returned None values — unexpected state after optimal solve."
            )

        return MPOResult(
            assets=moments.assets,
            planned_weights=weights_val,
            planned_trades=trades_val,
            initial_weights=current_weights,
            status=cast(SolverStatus, self.problem.status),
            objective_value=float(obj_val),
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
        full_inputs = {"moments": moments, **(inputs or {})}

        # Initial parameter setup (all handlers)
        for term, params in zip(self.objective.terms, self._term_params):
            get_objective_handler(term.spec).update(term.spec, params, full_inputs)

        # Identify cutting-plane CVaR terms that need iterative refinement
        cvar_terms = [
            (term, params)
            for term, params in zip(self.objective.terms, self._term_params)
            if isinstance(term.spec, CVaRCuttingPlane)
        ]

        weights_val: NDArray[np.floating] | None = None
        trades_val: NDArray[np.floating] | None = None
        obj_val: float | None = None

        for iteration in range(max_iter):
            self.problem.solve(enforce_dpp=True, warm_start=True, **solver_options)

            if self.problem.status not in {"optimal", "optimal_inaccurate"}:
                raise RuntimeError(
                    f"Iteration {iteration}: solve failed with {self.problem.status}"
                )

            weights_val = self.weights.value
            trades_val = self.trades.value
            obj_val = self.problem.value

            if not cvar_terms:
                break  # no cutting-plane CVaR → one-shot

            converged = True
            for term, params in cvar_terms:
                handler = get_refineable_handler(term.spec)
                assert weights_val is not None
                if not handler.refine(term.spec, params, weights_val, moments):
                    converged = False

            if converged:
                break
        else:
            total_cuts = sum(
                sum(params.get("cut_count", [0])) for _, params in cvar_terms
            )
            logger.warning(
                "CVaR cutting-plane did not converge in %d iterations "
                "(%d total cuts placed).",
                max_iter,
                total_cuts,
            )
        if weights_val is None or trades_val is None or obj_val is None:
            raise RuntimeError(
                "Solver returned None values — unexpected state after optimal solve."
            )

        return MPOResult(
            assets=moments.assets,
            planned_weights=weights_val,
            planned_trades=trades_val,
            initial_weights=current_weights,
            status=cast(SolverStatus, self.problem.status),
            objective_value=float(obj_val),
            solver_stats=self.problem.solver_stats,
        )
