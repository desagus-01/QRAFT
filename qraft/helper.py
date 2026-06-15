from __future__ import annotations

import cProfile
import pstats
from functools import wraps
from io import StringIO
from typing import Any, Callable

__all__ = ["profile"]


class profile:
    """Profile a function and print stats — works as decorator or context manager.

    Usage in notebooks::

        # As a decorator
        @profile
        def slow(x):
            return sum(range(x))

        slow(1_000_000)

        # With options
        @profile(sort="tottime", limit=10)
        def slow(x):
            return sum(range(x))

        slow(1_000_000)

        # As a context manager
        with profile():
            slow(1_000_000)

        with profile(sort="ncalls"):
            slow(1_000_000)
    """

    def __init__(
        self,
        func: Callable | None = None,
        *,
        sort: str = "cumtime",
        limit: int = 30,
    ):
        self.sort = sort
        self.limit = limit
        self.func = func
        self._profiler: cProfile.Profile | None = None

    def __enter__(self) -> profile:
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._profiler is not None:
            self._profiler.disable()
            _print_stats(self._profiler, sort=self.sort, limit=self.limit)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.func is not None:
            # Bare @profile applied to func during init
            return self._run(self.func, *args, **kwargs)

        # @profile(sort=..., limit=...) used as decorator factory
        # The function is passed as the first argument to __call__
        if args and callable(args[0]):
            f = args[0]

            @wraps(f)
            def wrapper(*a: Any, **kw: Any) -> Any:
                return self._run(f, *a, **kw)

            return wrapper

        msg = "profile() expects a callable when used as a decorator"
        raise TypeError(msg)

    def _run(self, f: Callable, *args: Any, **kwargs: Any) -> Any:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            return f(*args, **kwargs)
        finally:
            profiler.disable()
            _print_stats(profiler, sort=self.sort, limit=self.limit)


def _print_stats(profiler: cProfile.Profile, sort: str, limit: int) -> None:
    out = StringIO()
    ps = pstats.Stats(profiler, stream=out).sort_stats(sort)
    ps.print_stats(limit)
    print(out.getvalue())
