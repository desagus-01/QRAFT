from __future__ import annotations

import cProfile
import pstats
import tracemalloc
from functools import wraps
from io import StringIO
from pathlib import Path
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

        # Include Python allocation stats and write the report to a file
        with profile(memory=True, output_path="profile.txt"):
            slow(1_000_000)
    """

    def __init__(
        self,
        func: Callable | None = None,
        *,
        sort: str = "cumtime",
        limit: int = 30,
        memory: bool = False,
        memory_limit: int = 20,
        output_path: str | Path | None = None,
    ):
        self.sort = sort
        self.limit = limit
        self.memory = memory
        self.memory_limit = memory_limit
        self.output_path = output_path
        self.func = func
        self._profiler: cProfile.Profile | None = None

    def __enter__(self) -> profile:
        self._profiler = cProfile.Profile()
        if self.memory:
            tracemalloc.start(25)
        self._profiler.enable()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._profiler is not None:
            self._profiler.disable()
            report = _format_report(
                self._profiler,
                sort=self.sort,
                limit=self.limit,
                memory=self.memory,
                memory_limit=self.memory_limit,
            )
            _emit_report(report, self.output_path)

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
        if self.memory:
            tracemalloc.start(25)
        profiler.enable()
        try:
            return f(*args, **kwargs)
        finally:
            profiler.disable()
            report = _format_report(
                profiler,
                sort=self.sort,
                limit=self.limit,
                memory=self.memory,
                memory_limit=self.memory_limit,
            )
            _emit_report(report, self.output_path)


def _format_report(
    profiler: cProfile.Profile,
    sort: str,
    limit: int,
    *,
    memory: bool,
    memory_limit: int,
) -> str:
    out = StringIO()
    ps = pstats.Stats(profiler, stream=out).sort_stats(sort)
    ps.print_stats(limit)
    if memory:
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        out.write("\nMemory profile (Python allocations via tracemalloc)\n")
        out.write(f"Current allocated: {current / 1024 / 1024:.2f} MiB\n")
        out.write(f"Peak allocated:    {peak / 1024 / 1024:.2f} MiB\n")
        out.write(f"Top {memory_limit} allocation sites by line:\n")
        for stat in snapshot.statistics("lineno")[:memory_limit]:
            out.write(f"{stat}\n")
    return out.getvalue()


def _emit_report(report: str, output_path: str | Path | None) -> None:
    if output_path is None:
        print(report)
        return
    Path(output_path).write_text(report)
