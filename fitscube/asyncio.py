"""Utility functions for Fitscube."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
)

from tqdm.asyncio import tqdm

from fitscube.logging import TQDM_OUT

try:
    import uvloop

    USE_UVLOOP = True
except ImportError:
    USE_UVLOOP = False


class AsyncRunner(Protocol):
    """Protocol for async runner."""

    def __call__(self, main: Coroutine[Any, Any, T]) -> T: ...


if USE_UVLOOP:
    async_runner: AsyncRunner = uvloop.run
else:
    async_runner: AsyncRunner = asyncio.run  # type: ignore[no-redef]

P = ParamSpec("P")
T = TypeVar("T")


def sync_wrapper(coro: Callable[P, Coroutine[None, None, T]]) -> Callable[P, T]:
    @wraps(coro)
    def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        return async_runner(coro(*args, **kwargs))

    # Keep the function docs correct
    wrapper.__doc__ = coro.__doc__
    return wrapper


# Stolen from https://stackoverflow.com/a/61478547
async def gather_with_limit(
    limit: int | None, *coros: Awaitable[T], desc: str | None = None
) -> list[T]:
    """Gather with a limit on the number of coroutines running at once.

    Args:
        limit (int): The number of coroutines to run at once
        coros (Awaitable): The coroutines to run

    Returns:
        Awaitable: The result of the coroutines
    """
    if limit is None:
        return cast(list[T], await tqdm.gather(*coros, desc=desc, file=TQDM_OUT))

    semaphore = asyncio.Semaphore(limit)

    async def sem_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return cast(
        list[T],
        await tqdm.gather(*(sem_coro(c) for c in coros), desc=desc, file=TQDM_OUT),
    )
