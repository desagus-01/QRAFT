import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint as CvxConstraint
from numpy._typing import NDArray
from pydantic import validate_call

from globals import model_cfg
from scenarios.types import ConstraintDiag, ProbVector, View
from utils.helpers import select_operator, weighted_moments

logger = logging.getLogger(__name__)


def ens(prob_vector: ProbVector) -> int:
    """
    Effective number of scenarios (ENS) computed from a probability vector.

    ENS is defined as the exponential of the Shannon entropy of the
    discrete distribution represented by ``prob_vector``. The function
    validates that the result is in a sensible range and returns an integer
    approximation.

    Parameters
    ----------
    prob_vector : ProbVector
        Array-like probability vector summing to 1.

    Returns
    -------
    int
        Effective number of scenarios (rounded up).

    Raises
    ------
    RuntimeError
        If the computed ENS is outside the range [1, n_scenarios].
    """
    max = prob_vector.shape[0]

    ens = np.exp(-(np.sum(prob_vector * np.log(prob_vector))))

    if (ens < 1) or (ens > max):
        raise RuntimeError(
            "ENS is larger than total number of scenarios or smaller than 1."
        )

    return int(np.ceil(ens))  # ensure rounding is ok


# TODO: Finish writing this
def effective_rank(views_target):
    pass


@dataclass(frozen=True)
class CompiledView:
    """
    Compiled representation of a single :class:`View`.

    Stores the original view, the CVXPY constraint, and the raw ``lhs``/``rhs``
    expressions used to build it.  Keeping ``lhs`` and ``rhs`` here means
    diagnostic code never has to reverse-engineer the constraint math, and
    both construction and evaluation logic live in the same place.

    Attributes
    ----------
    view : View
        The original view specification.
    constraint : CvxConstraint
        The CVXPY constraint passed to the solver.
    lhs : cp.Expression
        Left-hand side of the constraint expression.
    rhs : cp.Expression | float | NDArray[np.floating]
        Right-hand side of the constraint expression.
    """

    view: View
    constraint: CvxConstraint
    lhs: cp.Expression
    rhs: cp.Expression | float | NDArray[np.floating]


def compile_view(view: View, posterior: cp.Variable, prior: ProbVector) -> CompiledView:
    """
    Build a :class:`CompiledView` from a single :class:`View`.

    Encapsulates all constraint-construction logic in one place so that
    :func:`_build_constraints` and :func:`get_constraints_diags` cannot
    diverge.

    Parameters
    ----------
    view : View
        View describing the constraint.
    posterior : cp.Variable
        CVXPY posterior-probability variable.
    prior : ProbVector
        Prior probability vector (reference for std/corr views).

    Returns
    -------
    CompiledView
        Frozen dataclass containing the constraint and its constituent
        ``lhs``/``rhs`` expressions.
    """
    operator_used = select_operator(view)

    match view.type:
        case "quantile":
            if view.views_target is None:
                raise ValueError("Your views_target cannot be None")
            lhs = view.data @ posterior
            rhs = view.views_target

        case "sorting":
            lhs = view.data[0] @ posterior
            rhs = view.data[1] @ posterior

        case "std":
            if view.views_target is None:
                raise ValueError("Your views_target cannot be None")
            mu_ref = view.data @ prior if view.mean_ref is None else view.mean_ref
            lhs = view.data**2 @ posterior
            rhs = view.views_target**2 + mu_ref**2

        case "mean":
            if view.views_target is None:
                raise ValueError("Your views_target cannot be None")
            lhs = view.data @ posterior
            rhs = view.views_target

        case "corr":
            # Anchor both mean and std on the prior
            mu_ref, std_ref = weighted_moments(view.data, prior)
            lhs = (view.data[0] * view.data[1]) @ posterior
            rhs = view.views_target * std_ref[0] * std_ref[1] + mu_ref[0] * mu_ref[1]

        case _:
            raise ValueError(f"Unsupported constraint type: {view.type}")

    constraint = operator_used(lhs, rhs)
    return CompiledView(view=view, constraint=constraint, lhs=lhs, rhs=rhs)


def diagnose_view(compiled: CompiledView) -> ConstraintDiag:
    """
    Produce a :class:`ConstraintDiag` for a compiled view after solving.

    ``active`` is defined uniformly across all view types as the L-inf
    residual ``max|lhs - rhs| <= 1e-5`` evaluated at the optimal posterior
    (available via CVXPY's ``.value`` attribute on each expression).

    Parameters
    ----------
    compiled : CompiledView
        A compiled view whose underlying CVXPY problem has already been solved.

    Returns
    -------
    ConstraintDiag
        Diagnostic dict with keys ``risk_driver``, ``sign``,
        ``constraint_value``, ``active``, and ``sensitivity``.
    """
    view = compiled.view
    lhs_val = np.asarray(compiled.lhs.value)
    rhs_val = (
        np.asarray(compiled.rhs.value)
        if isinstance(compiled.rhs, cp.Expression)
        else np.asarray(compiled.rhs)
    )
    active = bool(np.all(np.abs(lhs_val - rhs_val) <= 1e-5))
    dv = compiled.constraint.dual_value
    if dv is None:
        sensitivity: float | None = None
    else:
        raw = float(np.asarray(dv).flat[0])
        sensitivity = raw if view.sign_type != "equal_greater" else -raw
    return ConstraintDiag(
        risk_driver=view.risk_driver,
        sign=view.sign_type,
        constraint_value=view.views_target,
        active=active,
        sensitivity=sensitivity,
    )


def _build_constraints(
    views: list[View],
    posterior: cp.Variable,
    prior: ProbVector,
) -> tuple[list[CompiledView], list[CvxConstraint]]:
    """
    Compile views and assemble CVXPY constraints for the entropy pooling problem.

    Always includes the simplex constraints (``sum(posterior) == 1`` and
    ``posterior >= 0``) plus one constraint per view built via
    :func:`compile_view`.

    Parameters
    ----------
    views : list[View]
        List of views to incorporate.
    posterior : cp.Variable
        Posterior probability variable for the optimization.
    prior : ProbVector
        Prior probability vector used for reference values in some views.

    Returns
    -------
    tuple[list[CompiledView], list[CvxConstraint]]
        - ``compiled_views``: one :class:`CompiledView` per input view.
        - ``all_constraints``: full constraint list ready for ``cp.Problem``
          (view constraints + simplex constraints).
    """
    base: list[CvxConstraint] = [cp.sum(posterior) == 1, posterior >= 0]  # type: ignore[list-item]
    compiled_views: list[CompiledView] = [
        compile_view(view, posterior, prior) for view in views
    ]
    view_constraints: list[CvxConstraint] = [cv.constraint for cv in compiled_views]
    return compiled_views, view_constraints + base


# TODO: Fix info dict
def get_constraints_diags(
    compiled_views: list[CompiledView],
) -> list[ConstraintDiag]:
    """
    Produce diagnostic information for all views after solving EP.

    Delegates per-view evaluation to :func:`diagnose_view`, which uses the
    stored ``lhs``/``rhs`` expressions from each :class:`CompiledView`.
    No type-specific branching needed here.

    Parameters
    ----------
    compiled_views : list[CompiledView]
        Compiled views whose underlying CVXPY problem has been solved.

    Returns
    -------
    list[ConstraintDiag]
        Diagnostics dicts with keys ``risk_driver``, ``sign``,
        ``constraint_value``, ``active``, and ``sensitivity``.
    """
    return [diagnose_view(cv) for cv in compiled_views]


# TODO: Consider whether this is the best way (maybe instead of clipping give a v small value)
def clip_normalise_probs(prob: NDArray[np.floating]) -> ProbVector:
    """
    Clip tiny negative solver tolerances and renormalize a probability vector.

    Parameters
    ----------
    prob : NDArray[np.floating]
        Raw posterior returned by the solver, may contain small negative values
        due to numerical tolerances.

    Returns
    -------
    ProbVector
        Clipped and re-normalized posterior probabilities.

    Raises
    ------
    RuntimeError
        If clipping results in a zero-sum vector (posterior collapsed).
    """
    prob[prob < 0] = 0.0

    s = prob.sum()
    if s <= 0:
        raise RuntimeError("posterior collapsed after clipping (unexpected).")

    prob /= s
    return np.asarray(prob, dtype=float)


def entropy_pooling(
    prior: ProbVector,
    views: list[View],
    solver: str = "SCS",
    **solver_kwargs: str,
) -> ProbVector:
    """
    Apply Entropy Pooling (EP) to update a prior probability vector given views.

    The function sets up a convex optimisation minimizing KL divergence
    KL(posterior || prior) subject to linear constraints derived from
    ``views``. It returns a posterior probability vector if the problem
    solves to optimality.

    Parameters
    ----------
    prior : ProbVector
        Prior probability vector to be updated.
    views : list[View]
        List of views encoding linear constraints on expectations, quantiles,
        correlations or orderings.
    solver : str, optional
        CVXPY solver name to use (default: 'SCS').
    **solver_kwargs : dict
        Additional solver keyword arguments forwarded to cvxpy.

    Returns
    -------
    ProbVector
        Posterior probability vector satisfying the provided views.

    Raises
    ------
    RuntimeError
        If the CVXPY problem fails to solve to optimality or returns an
        infeasible/invalid posterior.
    """
    posterior = cp.Variable(prior.shape[0])
    compiled_views, all_constraints = _build_constraints(
        views=views, posterior=posterior, prior=prior
    )
    obj = cp.Minimize(cp.sum(cp.kl_div(posterior, prior)))
    prob = cp.Problem(obj, all_constraints)
    prob.solve(solver=solver, **solver_kwargs)
    posterior_res = posterior.value

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"EP did not solve optimally. status={prob.status}")
    if posterior_res is None:
        raise RuntimeError("Optimization failed or returned no solution!")
    if posterior_res.min() < -1e-6:
        raise RuntimeError(
            f"Materially negative posterior probability: min={posterior_res.min()}"
        )
    elif posterior_res.min() < 0:
        posterior_res = clip_normalise_probs(posterior_res)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "EP constraint diagnostics: %s",
            get_constraints_diags(compiled_views),
        )

    return posterior_res


@validate_call(config=model_cfg, validate_return=True)
def entropy_pooling_probs(
    prior: ProbVector,
    views: list[View],
    confidence: float = 1.0,
) -> ProbVector:
    """
    Run entropy pooling and blend the posterior with the prior by confidence.

    This wrapper computes the posterior using :func:`entropy_pooling` and
    then returns a convex combination ``confidence * posterior + (1-confidence) * prior``.

    Parameters
    ----------
    prior : ProbVector
        Prior probability vector.
    views : list[View]
        Views to impose through entropy pooling.
    confidence : float, optional
        Weight to give to the posterior in the convex combination (default: 1.0).

    Returns
    -------
    ProbVector
        Final probability vector after blending.
    """
    entropy_pooling_res = entropy_pooling(prior, views)

    return confidence * entropy_pooling_res + (1 - confidence) * prior
