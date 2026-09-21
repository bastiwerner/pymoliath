"""
.. include:: ../docs/io/README.md
   :start-after: ## AsyncIO
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
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, TypeSource]:
        """Runs the pipeline and resolves to the final TypeSource value."""
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
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncIO[TypeResult]:
            """Maps the applicative value's function, curried, over this AsyncIO's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncIO."""
        return f"AsyncIO({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncIO (same as __str__)."""
        return str(self)
