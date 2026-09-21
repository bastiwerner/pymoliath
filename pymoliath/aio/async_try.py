"""
.. include:: ../docs/try/README.md
   :start-after: ## AsyncTry
"""

from __future__ import annotations

from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    Generic,
    Tuple,
    TypeVar,
    Union,
)

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.exception import Failure, Success, Try
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class AsyncTry(Generic[TypeSource]):
    """Async Try Monad: a deferred computation which resolves to a Try[TypeSource] once awaited.

    Directly awaitable - nothing in a chain of map/bind/map_failure/... runs until the AsyncTry
    itself is awaited (`await an_async_try`), same lazy-pipeline design as AsyncMaybe
    (pymoliath/async_maybe.py), adapted for Try's two value-carrying channels (Success/Failure)
    instead of Maybe's value/no-value ones (Just/Nothing).

    Like the sync Try, map/bind/map_failure/bind_failure catch any exception raised by the
    callback (sync raise, or raised while awaiting an async callback's result) and turn it into
    a Failure, same as Success.map/Success.bind/Failure.map_failure/Failure.bind_failure do in
    pymoliath/exception.py. from_coroutine does not: like AsyncMaybe.from_coroutine, it only
    wraps the awaited result as Success, mirroring how the sync Success(value) constructor itself
    doesn't catch either - only the map/bind-family methods and the module-level `safe()` do.

    Callbacks passed to map/bind/map_failure/bind_failure/inspect/inspect_failure may be plain
    sync functions or `async def` functions - whichever is returned is auto-detected at the point
    it's called (awaited only if it actually is an awaitable), so real async I/O can be mixed
    freely with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Try[TypeSource]]]) -> None:
        """AsyncTry constructor which takes a zero-argument async callable resolving to a Try.

        Parameters
        ----------
        run: Callable[[], Awaitable[Try[TypeSource]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to a Try.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Try[TypeSource]]:
        """Runs the pipeline and resolves to the final Try[TypeSource]."""
        return self._run().__await__()

    @staticmethod
    def from_success(value: TypeSource) -> AsyncTry[TypeSource]:
        """Lifts a plain value into an already-Success AsyncTry.

        Parameters
        ----------
        value: TypeSource
            Value to be wrapped as Success once awaited.

        Returns
        -------
        async_try: AsyncTry[TypeSource]
        """

        async def run() -> Try[TypeSource]:
            """Resolves immediately to Success(value)."""
            return Success(value)

        return AsyncTry(run)

    @staticmethod
    def from_failure(exception: Exception) -> AsyncTry[TypeSource]:
        """Lifts an Exception into an already-Failure AsyncTry.

        Parameters
        ----------
        exception: Exception
            Exception to be wrapped as Failure once awaited.

        Returns
        -------
        async_try: AsyncTry[TypeSource]
        """

        async def run() -> Try[TypeSource]:
            """Resolves immediately to Failure(exception)."""
            return Failure(exception)

        return AsyncTry(run)

    @staticmethod
    def from_try(try_value: Try[TypeSource]) -> AsyncTry[TypeSource]:
        """Lifts an existing sync Try (Success or Failure) into an AsyncTry.

        Parameters
        ----------
        try_value: Try[TypeSource]
            Try to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_try: AsyncTry[TypeSource]
        """

        async def run() -> Try[TypeSource]:
            """Resolves immediately to `try_value`."""
            return try_value

        return AsyncTry(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[TypeSource]],
    ) -> AsyncTry[TypeSource]:
        """Wraps a zero-argument async callable producing a raw value as a Success once awaited.

        Does not catch exceptions raised by `coroutine_function` - only map/bind/map_failure/
        bind_failure do, matching how the sync Success(value) constructor itself doesn't catch.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[TypeSource]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncTry stays
            re-awaitable).

        Returns
        -------
        async_try: AsyncTry[TypeSource]
        """

        async def run() -> Try[TypeSource]:
            """Awaits `coroutine_function` and wraps its result as Success."""
            return Success(await coroutine_function())

        return AsyncTry(run)

    def map(
        self, function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncTry[TypeResult]:
        """AsyncTry functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function applied to the resolved value if Success.

        Returns
        -------
        async_try: AsyncTry[TypeResult]
            Returns a new AsyncTry which resolves to Success with the function result, to Failure
            if `function` raises, or to the original Failure unchanged without calling `function`,
            if this AsyncTry resolves to Failure.
        """

        async def run() -> Try[TypeResult]:
            """Awaits self, then applies `function`, short-circuiting on Failure."""
            outcome = await self
            match outcome:
                case Failure():
                    return outcome
                case Success(value):
                    try:
                        return Success(await _resolve(function(value)))
                    except Exception as e:
                        return Failure(e)

        return AsyncTry(run)

    def map_failure(
        self,
        function: Callable[[Exception], Union[Exception, Awaitable[Exception]]],
    ) -> AsyncTry[TypeSource]:
        """Calls `function` on the resolved Failure's exception, otherwise leaves Success untouched.

        Parameters
        ----------
        function: Callable[[Exception], Union[Exception, Awaitable[Exception]]]
            Sync or async function applied to the resolved exception if Failure.

        Returns
        -------
        async_try: AsyncTry[TypeSource]
            Returns a new AsyncTry which resolves to Failure with the function result, to Failure
            of any exception `function` itself raises, or to the original Success unchanged
            without calling `function`, if this AsyncTry resolves to Success.
        """

        async def run() -> Try[TypeSource]:
            """Awaits self, then applies `function` to the exception, short-circuiting on Success."""
            outcome = await self
            match outcome:
                case Success():
                    return outcome
                case Failure(exc):
                    try:
                        return Failure(await _resolve(function(exc)))
                    except Exception as e:
                        return Failure(e)

        return AsyncTry(run)

    def bind(
        self,
        function: Callable[
            [TypeSource],
            Union[AsyncTry[TypeResult], Try[TypeResult], Awaitable[Try[TypeResult]]],
        ],
    ) -> AsyncTry[TypeResult]:
        """AsyncTry bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], AsyncTry[TypeResult] | Try[TypeResult] | Awaitable[Try[TypeResult]]]
            Function applied to the resolved value if Success, returning another AsyncTry, a plain
            Try, or an awaitable resolving to a Try - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_try: AsyncTry[TypeResult]
            Returns a new AsyncTry with the function result if Success, a Failure of any exception
            `function` raises, or the original Failure unchanged without calling `function`.
        """

        async def run() -> Try[TypeResult]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Failure."""
            outcome = await self
            match outcome:
                case Failure():
                    return outcome
                case Success(value):
                    try:
                        result = function(value)
                        if isinstance(result, AsyncTry):
                            return await result
                        return await _resolve(result)
                    except Exception as e:
                        return Failure(e)

        return AsyncTry(run)

    def bind_failure(
        self,
        function: Callable[
            [Exception],
            Union[AsyncTry[TypeSource], Try[TypeSource], Awaitable[Try[TypeSource]]],
        ],
    ) -> AsyncTry[TypeSource]:
        """Calls `function` with the resolved Failure's exception, otherwise leaves Success untouched.

        Parameters
        ----------
        function: Callable[[Exception], AsyncTry[TypeSource] | Try[TypeSource] | Awaitable[Try[TypeSource]]]
            Function applied to the resolved exception if Failure, returning another AsyncTry, a
            plain Try, or an awaitable resolving to a Try - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_try: AsyncTry[TypeSource]
            Returns a new AsyncTry with the function result if Failure, a Failure of any exception
            `function` raises, or the original Success unchanged without calling `function`.
        """

        async def run() -> Try[TypeSource]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Success."""
            outcome = await self
            match outcome:
                case Success():
                    return outcome
                case Failure(exc):
                    try:
                        result = function(exc)
                        if isinstance(result, AsyncTry):
                            return await result
                        return await _resolve(result)
                    except Exception as e:
                        return Failure(e)

        return AsyncTry(run)

    def apply(
        self, applicative: AsyncTry[Callable[..., TypeResult]]
    ) -> AsyncTry[TypeResult]:
        """AsyncTry applicative interface for AsyncTrys containing a value (<*>).

        Parameters
        ----------
        applicative: AsyncTry[TypeApplicative] (TypeApplicative: any callable type)
            Applicative AsyncTry which contains a function and will be applied to the AsyncTry
            containing a value.

        Returns
        -------
        async_try: AsyncTry[TypeResult]
            Applies an AsyncTry containing a value of type TypeSource to an AsyncTry containing
            a function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncTry[TypeResult]:
            """Maps the applicative's function, curried, over this AsyncTry's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: AsyncTry[Callable[..., TypeResult]],
        applicative_value: AsyncTry[Any],
    ) -> AsyncTry[TypeResult]:
        """AsyncTry applicative interface for AsyncTrys containing a function (<*>).

        Parameters
        ----------
        applicative_value: AsyncTry[TypePure]
            AsyncTry value which will be applied to the AsyncTry containing a function.

        Returns
        -------
        async_try: AsyncTry[TypeResult]
            Applies an AsyncTry containing a function to an AsyncTry of type TypePure (value or
            function).
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncTry[TypeResult]:
            """Maps the curried applicative function, held by this AsyncTry, over `applicative_value`."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def and_(self, other: AsyncTry[TypeResult]) -> AsyncTry[TypeResult]:
        """Returns `other` if this AsyncTry resolves to Success, otherwise the original Failure.

        Parameters
        ----------
        other: AsyncTry[TypeResult]
            AsyncTry to be returned if this AsyncTry resolves to Success.

        Returns
        -------
        async_try: AsyncTry[TypeResult]
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncTry[TypeSource]) -> AsyncTry[TypeSource]:
        """Returns this AsyncTry if it resolves to Success, otherwise `other`.

        Parameters
        ----------
        other: AsyncTry[TypeSource]
            AsyncTry to be returned if this AsyncTry resolves to Failure.

        Returns
        -------
        async_try: AsyncTry[TypeSource]
        """

        async def run() -> Try[TypeSource]:
            """Awaits self, falling back to `other` if this AsyncTry resolves to Failure."""
            outcome = await self
            match outcome:
                case Success():
                    return outcome
                case Failure():
                    return await other

        return AsyncTry(run)

    def zip(self, other: AsyncTry[TypePure]) -> AsyncTry[Tuple[TypeSource, TypePure]]:
        """Combines this AsyncTry with another into an AsyncTry of a tuple, or Failure if either is Failure.

        Parameters
        ----------
        other: AsyncTry[TypePure]
            AsyncTry to be zipped with this AsyncTry.

        Returns
        -------
        async_try: AsyncTry[Tuple[TypeSource, TypePure]]
        """

        async def run() -> Try[Tuple[TypeSource, TypePure]]:
            """Awaits both self and `other`, combining their values if both are Success."""
            outcome = await self
            match outcome:
                case Failure():
                    return outcome
                case Success(value):
                    other_outcome = await other
                    return other_outcome.map(lambda o: (value, o))

        return AsyncTry(run)

    def flatten(self: AsyncTry[AsyncTry[TypeResult]]) -> AsyncTry[TypeResult]:
        """Flattens a nested AsyncTry by one level.

        Returns
        -------
        async_try: AsyncTry[TypeResult]
            Returns the nested AsyncTry's eventual result, or the original Failure without
            awaiting it, if this AsyncTry resolves to Failure.
        """

        async def run() -> Try[TypeResult]:
            """Awaits self, then awaits the nested AsyncTry if Success."""
            outcome = await self
            match outcome:
                case Failure():
                    return outcome
                case Success(inner):
                    return await inner

        return AsyncTry(run)

    def inspect(
        self, function: Callable[[TypeSource], Union[None, Awaitable[None]]]
    ) -> AsyncTry[TypeSource]:
        """Inspect the AsyncTry's resolved value of TypeSource.

        Parameters
        ----------
        function: Callable[[TypeSource], Union[None, Awaitable[None]]]
            Sync or async inspection function called with the resolved value if Success.

        Returns
        -------
        async_try: AsyncTry[TypeSource]
        """

        async def run() -> Try[TypeSource]:
            """Awaits self, calling `function` for its side effect only if Success."""
            outcome = await self
            match outcome:
                case Success(value):
                    await _resolve(function(value))
                case Failure():
                    pass
            return outcome

        return AsyncTry(run)

    def inspect_failure(
        self, function: Callable[[Exception], Union[None, Awaitable[None]]]
    ) -> AsyncTry[TypeSource]:
        """Inspect the AsyncTry's resolved Failure exception.

        Parameters
        ----------
        function: Callable[[Exception], Union[None, Awaitable[None]]]
            Sync or async inspection function called with the resolved exception if Failure.

        Returns
        -------
        async_try: AsyncTry[TypeSource]
        """

        async def run() -> Try[TypeSource]:
            """Awaits self, calling `function` for its side effect only if Failure."""
            outcome = await self
            match outcome:
                case Failure(exc):
                    await _resolve(function(exc))
                case Success():
                    pass
            return outcome

        return AsyncTry(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncTry."""
        return f"AsyncTry({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncTry (same as __str__)."""
        return str(self)
