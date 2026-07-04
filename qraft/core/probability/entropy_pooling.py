import logging
import operator
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint as CvxConstraint
from numpy.typing import NDArray
from pydantic import validate_call

from qraft.core.panel import ScenarioPanel
from qraft.core.probability.prob_vector import ProbVector
from qraft.core.scenarios.view_types import (
    ConstraintDiag,
    CorrView,
    EntropyPoolingResult,
    MeanView,
    QuantileView,
    RankingView,
    Sign,
    StdView,
    ViewDiagnostics,
    ViewSpec,
)
from qraft.globals import model_cfg
from qraft.utils.helpers import indicator_quantile_marginal, weighted_moments
from qraft.utils.log import warning_event

logger = logging.getLogger(__name__)
SUCCESS_STATUSES = ("optimal", "optimal_inaccurate")

_OPS: dict[Sign, Callable[[Any, Any], CvxConstraint]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
}


def ens(prob: ProbVector) -> float:
    p = prob[prob > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def _col(panel: ScenarioPanel, name: str) -> NDArray[np.floating]:
    """Resolve one column of scenario values from the panel."""
    return panel.values[name].to_numpy()


@dataclass(frozen=True)
class CompiledConstraint:
    """A single cvxpy constraint plus the metadata needed to diagnose it.

    ``risk_driver`` / ``sign`` / ``target`` are captured here at compile time
    so ``diagnose_view`` never needs to branch on the originating spec type
    (one ``RankingView`` produces several constraints, each with its own pair).
    """

    constraint: CvxConstraint
    lhs: cp.Expression
    rhs: cp.Expression | float | NDArray[np.floating]
    risk_driver: tuple[str, str] | str
    sign: Sign
    target: float | None


class InfeasibleViewsError(ValueError):
    pass


def compile_spec(
    spec: ViewSpec,
    panel: ScenarioPanel,
    posterior: cp.Variable,
    prior: ProbVector,
    mean_targets: dict[str, float],
) -> list[CompiledConstraint]:
    """Resolve a spec's data from ``panel`` and build its cvxpy constraint(s)."""
    match spec:
        case MeanView(asset, sign, target):
            x: NDArray[np.floating] = _col(panel, asset)
            lhs: cp.Expression = x @ posterior
            rhs: cp.Expression | float | NDArray[np.floating] = target
            return [
                CompiledConstraint(
                    constraint=_OPS[sign](lhs, rhs),
                    lhs=lhs,
                    rhs=rhs,
                    risk_driver=asset,
                    sign=sign,
                    target=target,
                )
            ]

        case StdView(asset, sign, target):
            x = _col(panel, asset)
            # anchor on a paired mean VIEW if one exists, else the prior mean
            mu_ref: float = mean_targets.get(asset, float(x @ prior))
            lhs = (x**2) @ posterior
            rhs = target**2 + mu_ref**2
            return [
                CompiledConstraint(
                    constraint=_OPS[sign](lhs, rhs),
                    lhs=lhs,
                    rhs=rhs,
                    risk_driver=asset,
                    sign=sign,
                    target=target,
                )
            ]

        case CorrView((a_name, b_name), sign, target):
            a: NDArray[np.floating] = _col(panel, a_name)
            b: NDArray[np.floating] = _col(panel, b_name)
            mu, sd = weighted_moments(np.vstack([a, b]), prior)
            lhs = (a * b) @ posterior
            rhs = target * sd[0] * sd[1] + mu[0] * mu[1]
            return [
                CompiledConstraint(
                    constraint=_OPS[sign](lhs, rhs),
                    lhs=lhs,
                    rhs=rhs,
                    risk_driver=(a_name, b_name),
                    sign=sign,
                    target=target,
                )
            ]

        case QuantileView(asset, quantile, target_prob):
            indicator = indicator_quantile_marginal(
                panel.values.select(asset), quantile, prob=prior
            )
            x = indicator.select("quant_ind").to_numpy().ravel()
            lhs = x @ posterior
            rhs = target_prob
            return [
                CompiledConstraint(
                    constraint=_OPS["<="](lhs, rhs),
                    lhs=lhs,
                    rhs=rhs,
                    risk_driver=asset,
                    sign="<=",
                    target=target_prob,
                )
            ]

        case RankingView(order):
            compiled: list[CompiledConstraint] = []
            for hi, lo in zip(order, order[1:]):  # hi >= lo for each adjacent pair
                lhs = _col(panel, hi) @ posterior
                rhs = _col(panel, lo) @ posterior
                compiled.append(
                    CompiledConstraint(
                        constraint=_OPS[">="](lhs, rhs),
                        lhs=lhs,
                        rhs=rhs,
                        risk_driver=(hi, lo),
                        sign=">=",
                        target=None,
                    )
                )
            return compiled

        case _:
            raise ValueError(f"Unsupported view spec: {type(spec).__name__}")


def _build_constraints(
    panel: ScenarioPanel,
    specs: Sequence[ViewSpec],
    posterior: cp.Variable,
    prior: ProbVector,
) -> tuple[list[CompiledConstraint], list[CvxConstraint]]:
    """Simplex constraints + one-or-more constraints per spec."""
    base: list[CvxConstraint] = [cp.sum(posterior) == 1, posterior >= 0]  # type: ignore[list-item]
    mean_targets: dict[str, float] = {
        v.asset: v.target for v in specs if isinstance(v, MeanView)
    }

    compiled: list[CompiledConstraint] = []
    for spec in specs:
        compiled += compile_spec(spec, panel, posterior, prior, mean_targets)

    return compiled, [c.constraint for c in compiled] + base


def diagnose_view(c: CompiledConstraint) -> ConstraintDiag:
    """Post-solve diagnostics for one compiled constraint (type-agnostic)."""
    lhs_val: NDArray[np.floating] = np.asarray(c.lhs.value)
    rhs_val: NDArray[np.floating] = (
        np.asarray(c.rhs.value)
        if isinstance(c.rhs, cp.Expression)
        else np.asarray(c.rhs)
    )
    active: bool = bool(np.all(np.abs(lhs_val - rhs_val) <= 1e-5))

    dual_value = c.constraint.dual_value
    sensitivity: float | None = (
        None
        if dual_value is None
        else float(np.asarray(dual_value).flat[0]) * (-1.0 if c.sign == ">=" else 1.0)
    )

    return ConstraintDiag(
        risk_driver=c.risk_driver,
        sign=c.sign,
        constraint_value=c.target,
        active=active,
        sensitivity=sensitivity,
    )


def get_constraints_diags(
    compiled: list[CompiledConstraint],
) -> tuple[ConstraintDiag, ...]:
    """Diagnostics for every compiled constraint after solving."""
    return tuple(diagnose_view(c) for c in compiled)


def _constraint_label(c: CompiledConstraint) -> str:
    return f"{c.risk_driver} {c.sign} {c.target}"


def _raise_infeasible_views(status: str, compiled: list[CompiledConstraint]) -> None:
    constraints = "; ".join(_constraint_label(c) for c in compiled) or "<none>"
    raise InfeasibleViewsError(
        f"Entropy pooling views are infeasible. status={status}; "
        f"constraints={constraints}. A correlation view below the prior's "
        "achievable minimum has no solution — relax the target or supply richer "
        "scenarios."
    )


# TODO: Consider whether this is the best way (maybe instead of clipping give a v small value)
def clip_normalise_probs(prob: NDArray[np.floating]) -> ProbVector:
    """Clip tiny negative solver tolerances and renormalise to sum to 1."""
    prob[prob < 0] = 0.0

    total: float = float(prob.sum())
    if total <= 0:
        raise RuntimeError("posterior collapsed after clipping (unexpected).")

    prob /= total
    return np.asarray(prob, dtype=float)


def entropy_pooling(
    panel: ScenarioPanel,
    specs: Sequence[ViewSpec],
    solver: str = "SCS",
    *,
    ens_warn_ratio: float = 0.1,
    **solver_kwargs: Any,
) -> EntropyPoolingResult:
    """Minimum-KL update of ``panel.prob`` subject to the supplied view specs."""
    prior: ProbVector = panel.prob
    posterior: cp.Variable = cp.Variable(prior.shape[0])
    ens_prior = ens(prior)

    compiled, all_constraints = _build_constraints(panel, specs, posterior, prior)

    problem: cp.Problem = cp.Problem(
        cp.Minimize(cp.sum(cp.kl_div(posterior, prior))), all_constraints
    )
    problem.solve(solver=solver, **solver_kwargs)

    if problem.status not in SUCCESS_STATUSES:
        _raise_infeasible_views(problem.status, compiled)

    posterior_value: NDArray[np.floating] | None = posterior.value
    if posterior_value is None:
        raise RuntimeError("Optimization failed or returned no solution!")
    if posterior_value.min() < -1e-6:
        raise RuntimeError(
            f"Materially negative posterior probability: min={posterior_value.min()}"
        )
    posterior_res: ProbVector = (
        clip_normalise_probs(posterior_value)
        if posterior_value.min() < 0
        else np.asarray(posterior_value, dtype=float)
    )

    ens_posterior = ens(posterior_res)
    ens_ratio = ens_posterior / ens_prior if ens_prior > 0 else 0.0
    ens_collapsed = ens_ratio < ens_warn_ratio
    if ens_collapsed:
        warning_event(
            logger,
            "views.ens_collapsed",
            "Entropy pooling posterior effective scenario count collapsed",
            ens_prior=ens_prior,
            ens_posterior=ens_posterior,
            ens_ratio=ens_ratio,
            ens_warn_ratio=ens_warn_ratio,
        )

    diagnostics = ViewDiagnostics(
        ens_prior=ens_prior,
        ens_posterior=ens_posterior,
        constraints=get_constraints_diags(compiled),
        solver_status=problem.status,
        ens_collapsed=ens_collapsed,
    )
    return EntropyPoolingResult(posterior=posterior_res, diagnostics=diagnostics)


@validate_call(config=model_cfg, validate_return=True)
def entropy_pooling_probs(
    panel: ScenarioPanel,
    specs: Sequence[ViewSpec],
    confidence: float = 1.0,
    solver: str = "SCS",
    ens_warn_ratio: float = 0.1,
    **solver_kwargs: Any,
) -> EntropyPoolingResult:
    """Run EP and linearly pool posterior with prior by ``confidence``.

    ``confidence`` is a linear opinion-pool weight, not Meucci's partial-view
    confidence/effective-number-of-views adjustment.
    """
    result = entropy_pooling(
        panel, specs, solver=solver, ens_warn_ratio=ens_warn_ratio, **solver_kwargs
    )
    posterior = confidence * result.posterior + (1.0 - confidence) * panel.prob
    return EntropyPoolingResult(posterior=posterior, diagnostics=result.diagnostics)
