import pytest

from qraft.construction.optimization.objectives.protocol import (
    get_objective_handler,
    get_refineable_handler,
    register_objective,
)


class LocalSpec:
    pass


class UnknownSpec:
    pass


def test_register_objective_registers_valid_handler() -> None:
    @register_objective(LocalSpec)
    class Handler:
        def allocate(self, spec, horizons, n_assets, n_scenarios=None):
            return {}

        def compile(self, spec, params, weights_at_horizon, trades_at_horizon, horizon):
            return 0, []

        def update(self, spec, params, inputs):
            return None

    assert get_objective_handler(LocalSpec()).__class__ is Handler


def test_register_objective_rejects_missing_method() -> None:
    class Spec:
        pass

    with pytest.raises(TypeError, match="missing method 'update'"):

        @register_objective(Spec)
        class BadHandler:
            def allocate(self, spec, horizons, n_assets, n_scenarios=None):
                return {}

            def compile(
                self, spec, params, weights_at_horizon, trades_at_horizon, horizon
            ):
                return 0, []


def test_register_objective_rejects_non_callable_method() -> None:
    class Spec:
        pass

    with pytest.raises(TypeError, match="exists but is not callable"):

        @register_objective(Spec)
        class BadHandler:
            allocate = 1

            def compile(
                self, spec, params, weights_at_horizon, trades_at_horizon, horizon
            ):
                return 0, []

            def update(self, spec, params, inputs):
                return None


def test_get_objective_handler_rejects_unknown_spec() -> None:
    with pytest.raises(TypeError, match="No objective handler registered"):
        get_objective_handler(UnknownSpec())


def test_get_refineable_handler_requires_refine_method() -> None:
    with pytest.raises(TypeError, match="does not implement 'refine'"):
        get_refineable_handler(LocalSpec())


def test_get_refineable_handler_accepts_refineable_handler() -> None:
    class RefineableSpec:
        pass

    @register_objective(RefineableSpec)
    class Handler:
        def allocate(self, spec, horizons, n_assets, n_scenarios=None):
            return {}

        def compile(self, spec, params, weights_at_horizon, trades_at_horizon, horizon):
            return 0, []

        def update(self, spec, params, inputs):
            return None

        def refine(self, spec, params, weights_val, optimizer_inputs):
            return True

    assert get_refineable_handler(RefineableSpec()).__class__ is Handler
