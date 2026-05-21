from typing import Any, Protocol

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray


class MPOObjectiveHandler(Protocol):
    def allocate(
        self, spec, horizons: int, n_assets: int, n_scenarios: int
    ) -> dict[str, cp.Parameter]:
        """
        Called ONCE at the start of a backtest.

        Create and return all CVXPY Parameter objects this term needs.
        Parameters are sized to the universe.

        Example return value:
          {"r_hat": cp.Parameter(n_assets)}
        """
        pass

    def compile(
        self,
        spec,
        params: dict[str, Any],
        weights_at_horizon: cp.Expression,
        trades_at_horizon: cp.Expression,
        horizon: int,
    ) -> tuple[cp.Expression, list[cp.Constraint]]:
        """
        Called ONCE after allocate()

        Build and return the CVXPY expression for this term.
        Must use cp.Parameter objects (not raw numpy) so the expression
        is DPP-compliant and can be reused across time steps.

        The expression should be written as something to MAXIMISE.
        """
        pass

    def update(
        self, spec, params: dict[str, cp.Parameter], inputs: dict[str, Any]
    ) -> None:
        pass


class RefineableMPOObjectiveHandler(MPOObjectiveHandler, Protocol):
    """Extension of MPOObjectiveHandler for handlers that support iterative
    cut refinement (e.g. cutting-plane CVaR approximations).
    """

    def refine(
        self,
        spec,
        params: dict[str, Any],
        weights_val: NDArray[np.floating],
        moments: Any,
    ) -> bool:
        """
        Called after each solve to tighten the approximation.

        Returns ``True`` when the solution has converged (no new cuts added),
        ``False`` when at least one cut was added and the problem should be
        re-solved.
        """
        ...


_REGISTRY: dict[type, MPOObjectiveHandler] = {}


def _validate_handler(handler: object) -> None:
    required = ("allocate", "compile", "update")

    for name in required:
        method = getattr(handler, name, None)

        if method is None:
            raise TypeError(
                f"{handler.__class__.__name__} is not a valid ObjectiveHandler: "
                f"missing method {name!r}."
            )

        if not callable(method):
            raise TypeError(
                f"{handler.__class__.__name__}.{name} exists but is not callable."
            )


def register_objective(spec_type: type):
    def decorator(handler_cls: type):
        handler = handler_cls()

        _validate_handler(handler)

        _REGISTRY[spec_type] = handler
        return handler_cls

    return decorator


def get_objective_handler(spec: object) -> MPOObjectiveHandler:
    handler = _REGISTRY.get(type(spec))

    if handler is None:
        raise TypeError(f"No objective handler registered for {type(spec).__name__}.")

    return handler


def get_refineable_handler(spec: object) -> RefineableMPOObjectiveHandler:
    handler = get_objective_handler(spec)

    if not callable(getattr(handler, "refine", None)):
        raise TypeError(
            f"The handler registered for {type(spec).__name__!r} "
            "does not implement 'refine'. Use a RefineableMPOObjectiveHandler."
        )

    return handler  # type: ignore[return-value]
