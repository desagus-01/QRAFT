import logging
import operator
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Sequence

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


class Anchor(NamedTuple):
    mu: float
    sigma2: float
    source: str


def resolve_anchors(
    specs: Sequence[ViewSpec], panel: ScenarioPanel, prior: ProbVector
) -> dict[str, Anchor]:
    """Resolve per-asset mean/variance anchors used by std/corr views."""
    mean_views: dict[str, MeanView] = {}
    anchored_assets: set[str] = set()

    for spec in specs:
        if isinstance(spec, MeanView):
            mean_views[spec.asset] = spec
        elif isinstance(spec, StdView):
            anchored_assets.add(spec.asset)
        elif isinstance(spec, CorrView):
            anchored_assets.update(spec.pair)

    anchors: dict[str, Anchor] = {}
    for asset in anchored_assets:
        x = _col(panel, asset)
        mu_prior, sd_prior = weighted_moments(x.reshape(1, -1), prior)
        mean_view = mean_views.get(asset)
        if mean_view is None:
            anchors[asset] = Anchor(
                float(mu_prior[0]), float(sd_prior[0] ** 2), "prior"
            )
        elif mean_view.sign == "==":
            anchors[asset] = Anchor((mean_view.target), float(sd_prior[0] ** 2), "view")
        else:
            raise ValueError(
                f"StdView on '{asset}' cannot be anchored to an inequality mean view; "
                "use sign '==' or drop one of the views."
            )

    return anchors


def compile_spec(
    spec: ViewSpec,
    panel: ScenarioPanel,
    posterior: cp.Variable,
    prior: ProbVector,
    anchors: dict[str, Anchor],
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
            mu_ref = anchors[asset].mu
            lhs = (x**2) @ posterior
            rhs = target**2 + mu_ref**2
            mean_lhs = x @ posterior
            return [
                CompiledConstraint(
                    constraint=_OPS[sign](lhs, rhs),
                    lhs=lhs,
                    rhs=rhs,
                    risk_driver=asset,
                    sign=sign,
                    target=target,
                ),
                CompiledConstraint(
                    constraint=_OPS["=="](mean_lhs, mu_ref),
                    lhs=mean_lhs,
                    rhs=mu_ref,
                    risk_driver=(asset, "anchor:mean"),
                    sign="==",
                    target=mu_ref,
                ),
            ]

        case CorrView((a_name, b_name), sign, target):
            a: NDArray[np.floating] = _col(panel, a_name)
            b: NDArray[np.floating] = _col(panel, b_name)
            a_anchor = anchors[a_name]
            b_anchor = anchors[b_name]
            a_mu = a_anchor.mu
            b_mu = b_anchor.mu
            a_second = a_anchor.sigma2 + a_mu**2
            b_second = b_anchor.sigma2 + b_mu**2
            lhs = (a * b) @ posterior
            rhs = (
                target * np.sqrt(a_anchor.sigma2) * np.sqrt(b_anchor.sigma2)
                + a_mu * b_mu
            )
            a_mean_lhs = a @ posterior
            b_mean_lhs = b @ posterior
            a_second_lhs = (a**2) @ posterior
            b_second_lhs = (b**2) @ posterior
            return [
                CompiledConstraint(
                    constraint=_OPS[sign](lhs, rhs),
                    lhs=lhs,
                    rhs=rhs,
                    risk_driver=(a_name, b_name),
                    sign=sign,
                    target=target,
                ),
                CompiledConstraint(
                    constraint=_OPS["=="](a_mean_lhs, a_mu),
                    lhs=a_mean_lhs,
                    rhs=a_mu,
                    risk_driver=(a_name, "anchor:mean"),
                    sign="==",
                    target=a_mu,
                ),
                CompiledConstraint(
                    constraint=_OPS["=="](b_mean_lhs, b_mu),
                    lhs=b_mean_lhs,
                    rhs=b_mu,
                    risk_driver=(b_name, "anchor:mean"),
                    sign="==",
                    target=b_mu,
                ),
                CompiledConstraint(
                    constraint=_OPS["=="](a_second_lhs, a_second),
                    lhs=a_second_lhs,
                    rhs=a_second,
                    risk_driver=(a_name, "anchor:second_moment"),
                    sign="==",
                    target=a_second,
                ),
                CompiledConstraint(
                    constraint=_OPS["=="](b_second_lhs, b_second),
                    lhs=b_second_lhs,
                    rhs=b_second,
                    risk_driver=(b_name, "anchor:second_moment"),
                    sign="==",
                    target=b_second,
                ),
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
    anchors = resolve_anchors(specs, panel, prior)

    compiled: list[CompiledConstraint] = []
    for spec in specs:
        compiled += compile_spec(spec, panel, posterior, prior, anchors)

    return compiled, [c.constraint for c in compiled] + base


def _constraint_value(
    value: cp.Expression | float | NDArray[np.floating],
) -> NDArray[np.floating]:
    return np.asarray(value.value if isinstance(value, cp.Expression) else value)


def _constraint_active(c: CompiledConstraint, prob: ProbVector | None = None) -> bool:
    if prob is None:
        lhs_val = _constraint_value(c.lhs)
        rhs_val = _constraint_value(c.rhs)
    else:
        lhs_val = _expression_at_prob(c.lhs, prob)
        rhs_val = _expression_at_prob(c.rhs, prob)
    return bool(np.all(np.abs(lhs_val - rhs_val) <= 1e-5))


def _expression_at_prob(
    expr: cp.Expression | float | NDArray[np.floating], prob: ProbVector
) -> NDArray[np.floating]:
    if not isinstance(expr, cp.Expression):
        return np.asarray(expr)

    variables = expr.variables()
    if len(variables) != 1:
        return _constraint_value(expr)

    original = variables[0].value
    variables[0].value = prob
    try:
        return np.asarray(expr.value)
    finally:
        variables[0].value = original


def diagnose_view(
    c: CompiledConstraint, prob: ProbVector | None = None
) -> ConstraintDiag:
    """Post-solve diagnostics for one compiled constraint (type-agnostic)."""
    active = _constraint_active(c, prob)

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
    prob: ProbVector | None = None,
) -> tuple[ConstraintDiag, ...]:
    """Diagnostics for every compiled constraint after solving."""
    return tuple(diagnose_view(c, prob) for c in compiled)


def _build_diagnostics(
    *,
    prior: ProbVector,
    posterior: ProbVector,
    compiled: list[CompiledConstraint],
    solver_status: str,
    ens_warn_ratio: float,
) -> ViewDiagnostics:
    ens_prior = ens(prior)
    ens_posterior = ens(posterior)
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

    return ViewDiagnostics(
        ens_prior=ens_prior,
        ens_posterior=ens_posterior,
        constraints=get_constraints_diags(compiled, posterior),
        solver_status=solver_status,
        ens_collapsed=ens_collapsed,
    )


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


@validate_call(config=model_cfg, validate_return=True)
def entropy_pooling(
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
    prior: ProbVector = panel.prob
    posterior_var: cp.Variable = cp.Variable(prior.shape[0])
    compiled, all_constraints = _build_constraints(panel, specs, posterior_var, prior)

    problem: cp.Problem = cp.Problem(
        cp.Minimize(cp.sum(cp.kl_div(posterior_var, prior))), all_constraints
    )
    problem.solve(solver=solver, **solver_kwargs)

    if problem.status not in SUCCESS_STATUSES:
        _raise_infeasible_views(problem.status, compiled)
    if problem.status == "optimal_inaccurate":
        logger.warning(
            "Entropy pooling solver returned optimal_inaccurate; results may not "
            "satisfy requested tolerances."
        )

    posterior_value: NDArray[np.floating] | None = posterior_var.value
    if posterior_value is None:
        raise RuntimeError("Optimization failed or returned no solution!")
    if posterior_value.min() < -1e-6:
        raise RuntimeError(
            f"Materially negative posterior probability: min={posterior_value.min()}"
        )
    raw_posterior: ProbVector = (
        clip_normalise_probs(posterior_value)
        if posterior_value.min() < 0
        else np.asarray(posterior_value, dtype=float)
    )
    posterior = confidence * raw_posterior + (1.0 - confidence) * prior
    diagnostics = _build_diagnostics(
        prior=prior,
        posterior=posterior,
        compiled=[],
        solver_status=problem.status,
        ens_warn_ratio=ens_warn_ratio,
    )
    diagnostics = ViewDiagnostics(
        ens_prior=diagnostics.ens_prior,
        ens_posterior=diagnostics.ens_posterior,
        constraints=get_constraints_diags(compiled, posterior),
        solver_status=diagnostics.solver_status,
        ens_collapsed=diagnostics.ens_collapsed,
    )
    return EntropyPoolingResult(posterior=posterior, diagnostics=diagnostics)
