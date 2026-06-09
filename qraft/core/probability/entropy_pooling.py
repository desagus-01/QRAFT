import logging
import operator
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cvxpy as cp
import numpy as np
from qraft.core.panel import ScenarioPanel
from cvxpy.constraints.constraint import Constraint as CvxConstraint
from qraft.core.probability.prob_vector import ProbVector
from qraft.core.scenarios.view_types import (
    ConstraintDiag,
    CorrView,
    MeanView,
    QuantileView,
    RankingView,
    Sign,
    StdView,
    ViewSpec,
)
from qraft.globals import model_cfg
from numpy.typing import NDArray
from pydantic import validate_call
from qraft.utils.helpers import indicator_quantile_marginal, weighted_moments

logger = logging.getLogger(__name__)

_OPS: dict[Sign, Callable[[Any, Any], CvxConstraint]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
}


def ens(prob_vector: ProbVector) -> int:
    """Effective number of scenarios: exp of the Shannon entropy."""
    max_scenarios: int = prob_vector.shape[0]
    value: float = float(np.exp(-(np.sum(prob_vector * np.log(prob_vector)))))

    if (value < 1) or (value > max_scenarios):
        raise RuntimeError(
            "ENS is larger than total number of scenarios or smaller than 1."
        )

    return int(np.ceil(value))


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
                panel.values.select(asset), quantile
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
) -> list[ConstraintDiag]:
    """Diagnostics for every compiled constraint after solving."""
    return [diagnose_view(c) for c in compiled]


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
    **solver_kwargs: Any,
) -> ProbVector:
    """Minimum-KL update of ``panel.prob`` subject to the supplied view specs."""
    prior: ProbVector = panel.prob
    posterior: cp.Variable = cp.Variable(prior.shape[0])

    compiled, all_constraints = _build_constraints(panel, specs, posterior, prior)

    problem: cp.Problem = cp.Problem(
        cp.Minimize(cp.sum(cp.kl_div(posterior, prior))), all_constraints
    )
    problem.solve(solver=solver, **solver_kwargs)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"EP did not solve optimally. status={problem.status}")

    posterior_res: NDArray[np.floating] | None = posterior.value
    if posterior_res is None:
        raise RuntimeError("Optimization failed or returned no solution!")
    if posterior_res.min() < -1e-6:
        raise RuntimeError(
            f"Materially negative posterior probability: min={posterior_res.min()}"
        )
    if posterior_res.min() < 0:
        posterior_res = clip_normalise_probs(posterior_res)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("EP constraint diagnostics: %s", get_constraints_diags(compiled))

    return posterior_res


@validate_call(config=model_cfg, validate_return=True)
def entropy_pooling_probs(
    panel: ScenarioPanel,
    specs: Sequence[ViewSpec],
    confidence: float = 1.0,
) -> ProbVector:
    """Run EP and blend posterior with prior by ``confidence``."""
    posterior: ProbVector = entropy_pooling(panel, specs)
    return confidence * posterior + (1.0 - confidence) * panel.prob
