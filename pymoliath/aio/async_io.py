"""
# AsyncIO

`AsyncIO` is the directly-awaitable counterpart of `pymoliath.io.IO` - see `pymoliath.aio` for the
general design shared by all `Async*` monads. Since `IO` has no failure state, `AsyncIO` resolves
directly to the plain value once awaited, rather than to a wrapper like `Just`/`Ok`.

```python
import asyncio

asyncio.run(AsyncIO(lambda: 10).map(lambda x: x + 1))  # 11


async def fetch(x: int) -> int: ...


asyncio.run(AsyncIO.from_value(10).map(fetch))  # async callback, auto-detected
asyncio.run(AsyncIO.from_value(10).bind(lambda x: AsyncIO.from_value(x + 1)))  # 11


async def fetch_ten() -> int:
    return 10


asyncio.run(AsyncIO.from_coroutine(fetch_ten))  # 10
```
"""

from __future__ import annotations

from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    Generic,
    TypeVar,
    Union,
)

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.io import IO
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class AsyncIO(Generic[TypeSource]):
    """Async IO Monad: a deferred computation which resolves to a raw TypeSource value once awaited.

    Directly awaitable - nothing in a chain of map/bind/apply/... runs until the AsyncIO itself is
    awaited (`await an_async_io`), mirroring AsyncMaybe (pymoliath/async_maybe.py) but for a monad
    with no failure state: IO always "succeeds", so awaiting resolves directly to the plain
    TypeSource value, exactly like sync IO.run() returns TypeSource directly rather than a wrapper.

    Callbacks passed to map/bind may be plain sync functions or `async def` functions - whichever is
    returned is auto-detected at the point it's called (awaited only if it actually is an awaitable),
    so real async I/O can be mixed freely with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[TypeSource]]) -> None:
        """AsyncIO constructor which takes a zero-argument async callable resolving to a value.

        Parameters
        ----------
        run: Callable[[], Awaitable[TypeSource]]
            Zero-argument callable returning a fresh awaitable each call, resolving to the value.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.

        Examples
        --------
        >>> import asyncio
        >>> async def run(): return 10
        >>> asyncio.run(AsyncIO(run))
        10
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, TypeSource]:
        """Runs the pipeline and resolves to the final TypeSource value.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncIO.from_value(10))
        10
        """
        return self._run().__await__()

    @staticmethod
    def from_value(value: TypeSource) -> AsyncIO[TypeSource]:
        """Lifts a plain, already-resolved value into an AsyncIO.

        Parameters
        ----------
        value: TypeSource
            Value to be resolved to once awaited.

        Returns
        -------
        async_io: AsyncIO[TypeSource]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncIO.from_value(10))
        10
        """

        async def run() -> TypeSource:
            """Resolves immediately to `value`."""
            return value

        return AsyncIO(run)

    @staticmethod
    def from_io(io: IO[TypeSource]) -> AsyncIO[TypeSource]:
        """Lifts an existing sync IO into an AsyncIO, running it lazily once awaited.

        Parameters
        ----------
        io: IO[TypeSource]
            IO to be run inside the AsyncIO's deferred pipeline.

        Returns
        -------
        async_io: AsyncIO[TypeSource]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncIO.from_io(IO(lambda: 10)))
        10
        """

        async def run() -> TypeSource:
            """Runs `io` and resolves to its result."""
            return io.run()

        return AsyncIO(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[TypeSource]],
    ) -> AsyncIO[TypeSource]:
        """Wraps a zero-argument async callable producing the value once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[TypeSource]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncIO stays re-awaitable).

        Returns
        -------
        async_io: AsyncIO[TypeSource]

        Examples
        --------
        >>> import asyncio
        >>> async def fetch_ten() -> int: return 10
        >>> asyncio.run(AsyncIO.from_coroutine(fetch_ten))
        10
        """

        async def run() -> TypeSource:
            """Awaits `coroutine_function` and resolves to its result."""
            return await coroutine_function()

        return AsyncIO(run)

    def map(
        self, function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncIO[TypeResult]:
        """AsyncIO functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function applied to the resolved value.

        Returns
        -------
        async_io: AsyncIO[TypeResult]
            Returns a new AsyncIO which resolves to the function result.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncIO.from_value(5).map(lambda x: x + 1))
        6
        """

        async def run() -> TypeResult:
            """Awaits self, then applies `function`."""
            return await _resolve(function(await self))

        return AsyncIO(run)

    def bind(
        self,
        function: Callable[
            [TypeSource],
            Union[AsyncIO[TypeResult], IO[TypeResult], Awaitable[TypeResult]],
        ],
    ) -> AsyncIO[TypeResult]:
        """AsyncIO bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], AsyncIO[TypeResult] | IO[TypeResult] | Awaitable[TypeResult]]
            Function applied to the resolved value, returning another AsyncIO, a plain IO, or an
            awaitable resolving to the value - whichever shape is returned is auto-detected.

        Returns
        -------
        async_io: AsyncIO[TypeResult]
            Returns a new AsyncIO with the function result.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncIO.from_value(5).bind(lambda x: AsyncIO.from_value(x + 1)))
        6
        """

        async def run() -> TypeResult:
            """Awaits self, then chains into `function`'s result - fully deferred, unlike sync
            IO.bind() which runs self eagerly."""
            result = function(await self)
            if isinstance(result, AsyncIO):
                return await result
            if isinstance(result, IO):
                return result.run()
            return await _resolve(result)

        return AsyncIO(run)

    def apply(
        self, applicative: AsyncIO[Callable[..., TypeResult]]
    ) -> AsyncIO[TypeResult]:
        """AsyncIO applicative interface for AsyncIOs containing a value (<*>).

        Parameters
        ----------
        applicative: AsyncIO[TypeApplicative] (TypeApplicative: any callable type)
            Applicative AsyncIO which contains a function and will be applied to the AsyncIO
            containing a value.

        Returns
        -------
        async_io: AsyncIO[TypeResult]
            Applies an AsyncIO containing a value of type TypeSource to an AsyncIO containing a
            function.

        Examples
        --------
        >>> import asyncio
        >>> val = AsyncIO.from_value(10)
        >>> func = AsyncIO.from_value(lambda x: x * 2)
        >>> asyncio.run(val.apply(func))
        20
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncIO[TypeResult]:
            """Maps the applicative's function, curried, over this AsyncIO's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: AsyncIO[Callable[..., TypeResult]], applicative_value: AsyncIO[Any]
    ) -> AsyncIO[TypeResult]:
        """AsyncIO applicative interface for AsyncIOs containing a function (<*>).

        Parameters
        ----------
        applicative_value: AsyncIO[TypePure]
            AsyncIO value which will be applied to the AsyncIO containing a function.

        Returns
        -------
        async_io: AsyncIO[TypeResult]
            Applies an AsyncIO containing a function to an AsyncIO of type TypePure (value or
            function).

        Examples
        --------
        >>> import asyncio
        >>> func = AsyncIO.from_value(lambda x: x * 2)
        >>> val = AsyncIO.from_value(10)
        >>> asyncio.run(func.apply2(val))
        20
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncIO[TypeResult]:
            """Maps the applicative value's function, curried, over this AsyncIO's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncIO.

        Examples
        --------
        >>> str(AsyncIO.from_value(10))  # doctest: +ELLIPSIS
        'AsyncIO(<function...>)'
        """
        return f"AsyncIO({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncIO (same as __str__).

        Examples
        --------
        >>> repr(AsyncIO.from_value(10))  # doctest: +ELLIPSIS
        'AsyncIO(<function...>)'
        """
        return str(self)
