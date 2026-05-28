from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from scenarios.types import ProbVector
from time_series.estimation import OLSResults, weighted_ols


@dataclass(frozen=True, slots=True)
class StepResult:
    step: int
    feature_added: str
    score: float
    criterion: str

    def __repr__(self) -> str:
        return (
            f"Step {self.step}: +'{self.feature_added}' "
            f"({self.criterion}={self.score:.6f})"
        )


@dataclass(frozen=True, slots=True)
class ForwardRegressionResult:
    selected_features: list[str]
    final_model: OLSResults
    steps: list[StepResult] = field(default_factory=list)

    @property
    def n_selected(self) -> int:
        return len(self.selected_features)


Criterion = Literal["bic", "aic", "r2", "p_value"]


def _score_candidate(res: OLSResults, criterion: Criterion) -> float:
    if criterion == "aic":
        return res.aic
    if criterion == "bic":
        return res.bic
    if criterion == "r2":
        return res.r_squared
    return float(res.p_values[-1, 0])


def _is_improvement(new_score: float, best_score: float, criterion: Criterion) -> bool:
    if criterion in ("aic", "bic"):
        return new_score < best_score
    if criterion == "r2":
        return new_score > best_score
    return True


def _pick_best_candidate(
    scores: dict[int, float], criterion: Criterion
) -> tuple[int, float]:
    if criterion in ("aic", "bic", "p_value"):
        best = min(scores, key=scores.get)  # type: ignore[arg-type]
    else:
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best, scores[best]


def _prune_insignificant(
    dependent_var: NDArray[np.floating],
    independent_vars: NDArray[np.floating],
    feature_names: list[str],
    included_idx: list[int],
    p_value_threshold: float,
    prob: ProbVector | None,
) -> list[int]:
    """Re-fit the model and iteratively drop the worst feature whose p-value
    exceeds *p_value_threshold*."""
    changed = True
    while changed:
        changed = False
        if not included_idx:
            break
        included_features = independent_vars[:, included_idx]
        names = [feature_names[j] for j in included_idx]
        res = weighted_ols(
            dependent_var, included_features, feature_names=names, prob=prob
        )
        p_vals = res.p_values.ravel()
        worst_pos = int(np.argmax(p_vals))
        if p_vals[worst_pos] > p_value_threshold:
            included_idx.pop(worst_pos)
            changed = True
    return included_idx


def _evaluate_candidates(
    dependent_var: NDArray[np.floating],
    independent_vars: NDArray[np.floating],
    feature_names: list[str],
    included_idx: list[int],
    remaining: list[int],
    criterion: Criterion,
    prob: ProbVector | None,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for i in remaining:
        trial_idx = included_idx + [i]
        feature_trial = independent_vars[:, trial_idx]
        names = [feature_names[j] for j in trial_idx]
        res = weighted_ols(dependent_var, feature_trial, feature_names=names, prob=prob)
        scores[i] = _score_candidate(res, criterion)
    return scores


def _fit_final_model(
    dependent_var: NDArray[np.floating],
    independent_vars: NDArray[np.floating],
    feature_names: list[str],
    included_idx: list[int],
    prob: ProbVector | None,
) -> OLSResults:
    best_features = independent_vars[:, included_idx]
    names_final = [feature_names[i] for i in included_idx]
    return weighted_ols(
        dependent_var, best_features, feature_names=names_final, prob=prob
    )


def forward_regression(
    dependent_var: NDArray[np.floating],
    independent_vars: NDArray[np.floating],
    feature_names: list[str],
    criterion: Criterion,
    p_value_threshold: float = 0.05,
    prob: ProbVector | None = None,
) -> ForwardRegressionResult:
    included_idx: list[int] = []
    n_features = independent_vars.shape[1]
    best_score = np.inf if criterion in ("aic", "bic") else -np.inf
    steps: list[StepResult] = []

    while True:
        remaining = [i for i in range(n_features) if i not in included_idx]
        if not remaining:
            break

        scores = _evaluate_candidates(
            dependent_var,
            independent_vars,
            feature_names,
            included_idx,
            remaining,
            criterion,
            prob,
        )

        best_i, new_score = _pick_best_candidate(scores, criterion)

        if criterion == "p_value":
            if new_score > p_value_threshold:
                break
            included_idx.append(best_i)
            steps.append(
                StepResult(
                    step=len(steps) + 1,
                    feature_added=feature_names[best_i],
                    score=new_score,
                    criterion=criterion,
                )
            )
            included_idx = _prune_insignificant(
                dependent_var,
                independent_vars,
                feature_names,
                included_idx,
                p_value_threshold,
                prob,
            )
        else:
            if not _is_improvement(new_score, best_score, criterion):
                break
            best_score = new_score
            included_idx.append(best_i)
            steps.append(
                StepResult(
                    step=len(steps) + 1,
                    feature_added=feature_names[best_i],
                    score=new_score,
                    criterion=criterion,
                )
            )

    if not included_idx:
        raise ValueError(
            f"No feature improved the model under criterion='{criterion}' "
            f"(p_value_threshold={p_value_threshold})"
        )

    final_model = _fit_final_model(
        dependent_var,
        independent_vars,
        feature_names,
        included_idx,
        prob,
    )

    return ForwardRegressionResult(
        selected_features=[feature_names[i] for i in included_idx],
        final_model=final_model,
        steps=steps,
    )
