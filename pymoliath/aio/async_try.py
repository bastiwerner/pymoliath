"""
# AsyncTry

`AsyncTry` is the directly-awaitable counterpart of `pymoliath.exception.Try` - see
`pymoliath.aio` for the general design shared by all `Async*` monads.

```python
import asyncio

asyncio.run(AsyncTry.from_success(10).map(lambda x: x + 1))  # Success(11)
asyncio.run(AsyncTry.from_success(10).map(lambda x: 1 / 0))  # Failure(...), exception caught


async def fetch(x: int) -> int: ...


asyncio.run(AsyncTry.from_success(10).map(fetch))  # async callback, auto-detected
asyncio.run(AsyncTry.from_success(10).bind(lambda x: AsyncTry.from_success(x + 1)))  # Success(11)


async def fetch_ten() -> int:
    return 10


asyncio.run(AsyncTry.from_coroutine(fetch_ten))  # Success(10)
```
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from typing import Any, Generic, TypeVar, assert_never

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.exception import Failure, Success, Try

T = TypeVar("T")
U = TypeVar("U")

# Function-scoped TypeVar for the static constructors: the class-scoped T would be Unknown when
# called on the unspecialized class (`AsyncTry.from_success(1)`).
V = TypeVar("V")
# Function-scoped TypeVars for `map2`.
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


class AsyncTry(Generic[T]):
    """Async Try Monad: a deferred computation which resolves to a Try[T] once awaited.

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

    def __init__(self, run: Callable[[], Awaitable[Try[T]]]) -> None:
        """AsyncTry constructor which takes a zero-argument async callable resolving to a Try.

        Parameters
        ----------
        run: Callable[[], Awaitable[Try[T]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to a Try.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.

        Examples
        --------
        >>> import asyncio
        >>> async def run(): return Success(10)
        >>> asyncio.run(AsyncTry(run))
        Success(10)
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Try[T]]:
        """Runs the pipeline and resolves to the final Try[T].

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(10))
        Success(10)
        """
        return self._run().__await__()

    @staticmethod
    def from_success(value: V) -> AsyncTry[V]:
        """Lifts a plain value into an already-Success AsyncTry.

        Parameters
        ----------
        value: V
            Value to be wrapped as Success once awaited.

        Returns
        -------
        async_try: AsyncTry[V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(10))
        Success(10)
        """

        async def run() -> Try[V]:
            """Resolves immediately to Success(value)."""
            return Success(value)

        return AsyncTry(run)

    @staticmethod
    # V is deliberately only in the return type: like a bare Failure, the Success type is left open
    # to be solved from context.
    def from_failure(exception: Exception) -> AsyncTry[V]:  # pyright: ignore[reportInvalidTypeVarUse]
        """Lifts an Exception into an already-Failure AsyncTry.

        Parameters
        ----------
        exception: Exception
            Exception to be wrapped as Failure once awaited.

        Returns
        -------
        async_try: AsyncTry[V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")))
        Failure(ValueError('boom'))
        """

        async def run() -> Try[V]:
            """Resolves immediately to Failure(exception)."""
            return Failure(exception)

        return AsyncTry(run)

    @staticmethod
    def from_try(try_value: Try[V]) -> AsyncTry[V]:
        """Lifts an existing sync Try (Success or Failure) into an AsyncTry.

        Parameters
        ----------
        try_value: Try[V]
            Try to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_try: AsyncTry[V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_try(Success(10)))
        Success(10)
        >>> asyncio.run(AsyncTry.from_try(Failure(ValueError("boom"))))
        Failure(ValueError('boom'))
        """

        async def run() -> Try[V]:
            """Resolves immediately to `try_value`."""
            return try_value

        return AsyncTry(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[V]],
    ) -> AsyncTry[V]:
        """Wraps a zero-argument async callable producing a raw value as a Success once awaited.

        Does not catch exceptions raised by `coroutine_function` - only map/bind/map_failure/
        bind_failure do, matching how the sync Success(value) constructor itself doesn't catch.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[V]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncTry stays
            re-awaitable).

        Returns
        -------
        async_try: AsyncTry[V]

        Examples
        --------
        >>> import asyncio
        >>> async def fetch_ten() -> int: return 10
        >>> asyncio.run(AsyncTry.from_coroutine(fetch_ten))
        Success(10)
        """

        async def run() -> Try[V]:
            """Awaits `coroutine_function` and wraps its result as Success."""
            return Success(await coroutine_function())

        return AsyncTry(run)

    def map(self, function: Callable[[T], U | Awaitable[U]]) -> AsyncTry[U]:
        """AsyncTry functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[T], U | Awaitable[U]]
            Sync or async function applied to the resolved value if Success.

        Returns
        -------
        async_try: AsyncTry[U]
            Returns a new AsyncTry which resolves to Success with the function result, to Failure
            if `function` raises, or to the original Failure unchanged without calling `function`,
            if this AsyncTry resolves to Failure.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(5).map(lambda x: x + 1))
        Success(6)
        >>> def boom(x: int) -> int:
        ...     raise ValueError("boom")
        >>> asyncio.run(AsyncTry.from_success(5).map(boom))
        Failure(ValueError('boom'))
        """

        async def run() -> Try[U]:
            """Awaits self, then applies `function`, short-circuiting on Failure."""
            match outcome := await self:
                case Failure(exception):
                    return Failure(exception)
                case Success(value):
                    try:
                        return Success(await _resolve(function(value)))
                    except Exception as e:
                        return Failure(e)
                case _:
                    assert_never(outcome)

        return AsyncTry(run)

    def map_failure(
        self,
        function: Callable[[Exception], Exception | Awaitable[Exception]],
    ) -> AsyncTry[T]:
        """Calls `function` on the resolved Failure's exception, otherwise leaves Success untouched.

        Parameters
        ----------
        function: Callable[[Exception], Exception | Awaitable[Exception]]
            Sync or async function applied to the resolved exception if Failure.

        Returns
        -------
        async_try: AsyncTry[T]
            Returns a new AsyncTry which resolves to Failure with the function result, to Failure
            of any exception `function` itself raises, or to the original Success unchanged
            without calling `function`, if this AsyncTry resolves to Success.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).map_failure(lambda e: TypeError(str(e))))
        Failure(TypeError('boom'))
        >>> asyncio.run(AsyncTry.from_success(10).map_failure(lambda e: TypeError(str(e))))
        Success(10)
        """

        async def run() -> Try[T]:
            """Awaits self, then applies `function` to the exception, short-circuiting on Success."""
            match outcome := await self:
                case Success(value):
                    return Success(value)
                case Failure(exc):
                    try:
                        return Failure(await _resolve(function(exc)))
                    except Exception as e:
                        return Failure(e)
                case _:
                    assert_never(outcome)

        return AsyncTry(run)

    def bind(
        self,
        function: Callable[
            [T],
            AsyncTry[U] | Try[U] | Awaitable[Try[U]],
        ],
    ) -> AsyncTry[U]:
        """AsyncTry bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[T], AsyncTry[U] | Try[U] | Awaitable[Try[U]]]
            Function applied to the resolved value if Success, returning another AsyncTry, a plain
            Try, or an awaitable resolving to a Try - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_try: AsyncTry[U]
            Returns a new AsyncTry with the function result if Success, a Failure of any exception
            `function` raises, or the original Failure unchanged without calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(5).bind(lambda x: AsyncTry.from_success(x + 1)))
        Success(6)
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).bind(lambda x: AsyncTry.from_success(x + 1)))
        Failure(ValueError('boom'))
        """

        async def run() -> Try[U]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Failure."""
            match outcome := await self:
                case Failure(exception):
                    return Failure(exception)
                case Success(value):
                    try:
                        result = function(value)
                        if isinstance(result, AsyncTry):
                            return await result
                        return await _resolve(result)
                    except Exception as e:
                        return Failure(e)
                case _:
                    assert_never(outcome)

        return AsyncTry(run)

    def bind_failure(
        self,
        function: Callable[
            [Exception],
            AsyncTry[T] | Try[T] | Awaitable[Try[T]],
        ],
    ) -> AsyncTry[T]:
        """Calls `function` with the resolved Failure's exception, otherwise leaves Success untouched.

        Parameters
        ----------
        function: Callable[[Exception], AsyncTry[T] | Try[T] | Awaitable[Try[T]]]
            Function applied to the resolved exception if Failure, returning another AsyncTry, a
            plain Try, or an awaitable resolving to a Try - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_try: AsyncTry[T]
            Returns a new AsyncTry with the function result if Failure, a Failure of any exception
            `function` raises, or the original Success unchanged without calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).bind_failure(lambda e: AsyncTry.from_success(0)))
        Success(0)
        >>> asyncio.run(AsyncTry.from_success(10).bind_failure(lambda e: AsyncTry.from_success(0)))
        Success(10)
        """

        async def run() -> Try[T]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Success."""
            match outcome := await self:
                case Success(value):
                    return Success(value)
                case Failure(exc):
                    try:
                        result = function(exc)
                        if isinstance(result, AsyncTry):
                            return await result
                        return await _resolve(result)
                    except Exception as e:
                        return Failure(e)
                case _:
                    assert_never(outcome)

        return AsyncTry(run)

    def apply(self, function: AsyncTry[Callable[[T], U]]) -> AsyncTry[U]:
        """Applies the function wrapped in `function` to this AsyncTry's value (<*>).

        If both fail, the Failure of `function` takes precedence, as in `Try.apply`. For functions of
        several arguments, use the module-level `map2`.

        Parameters
        ----------
        function: AsyncTry[Callable[[T], U]]
            AsyncTry which contains a function of one argument.

        Returns
        -------
        async_try: AsyncTry[U]
            Returns the function applied to this AsyncTry's value.

        Examples
        --------
        >>> import asyncio
        >>> val = AsyncTry.from_success(10)
        >>> func = AsyncTry.from_success(lambda x: x * 2)
        >>> asyncio.run(val.apply(func))
        Success(20)
        """
        return function.bind(lambda inner: self.map(inner))

    def and_(self, other: AsyncTry[U]) -> AsyncTry[U]:
        """Returns `other` if this AsyncTry resolves to Success, otherwise the original Failure.

        Parameters
        ----------
        other: AsyncTry[U]
            AsyncTry to be returned if this AsyncTry resolves to Success.

        Returns
        -------
        async_try: AsyncTry[U]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(1).and_(AsyncTry.from_success(2)))
        Success(2)
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).and_(AsyncTry.from_success(2)))
        Failure(ValueError('boom'))
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncTry[T]) -> AsyncTry[T]:
        """Returns this AsyncTry if it resolves to Success, otherwise `other`.

        Parameters
        ----------
        other: AsyncTry[T]
            AsyncTry to be returned if this AsyncTry resolves to Failure.

        Returns
        -------
        async_try: AsyncTry[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(1).or_(AsyncTry.from_success(2)))
        Success(1)
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).or_(AsyncTry.from_success(2)))
        Success(2)
        """

        async def run() -> Try[T]:
            """Awaits self, falling back to `other` if this AsyncTry resolves to Failure."""
            match outcome := await self:
                case Success(value):
                    return Success(value)
                case Failure():
                    return await other
                case _:
                    assert_never(outcome)

        return AsyncTry(run)

    def zip(self, other: AsyncTry[U]) -> AsyncTry[tuple[T, U]]:
        """Combines this AsyncTry with another into an AsyncTry of a tuple, or Failure if either is Failure.

        Parameters
        ----------
        other: AsyncTry[U]
            AsyncTry to be zipped with this AsyncTry.

        Returns
        -------
        async_try: AsyncTry[tuple[T, U]]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(1).zip(AsyncTry.from_success(2)))
        Success((1, 2))
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).zip(AsyncTry.from_success(2)))
        Failure(ValueError('boom'))
        """

        async def run() -> Try[tuple[T, U]]:
            """Awaits both self and `other`, combining their values if both are Success."""
            match outcome := await self:
                case Failure(exception):
                    return Failure(exception)
                case Success(value):
                    other_outcome = await other
                    return other_outcome.map(lambda o: (value, o))
                case _:
                    assert_never(outcome)

        return AsyncTry(run)

    def flatten(self: AsyncTry[AsyncTry[U]]) -> AsyncTry[U]:
        """Flattens a nested AsyncTry by one level.

        Returns
        -------
        async_try: AsyncTry[U]
            Returns the nested AsyncTry's eventual result, or the original Failure without
            awaiting it, if this AsyncTry resolves to Failure.

        Examples
        --------
        >>> import asyncio
        >>> nested = AsyncTry.from_success(AsyncTry.from_success(1))
        >>> asyncio.run(nested.flatten())
        Success(1)
        """

        async def run() -> Try[U]:
            """Awaits self, then awaits the nested AsyncTry if Success."""
            match outcome := await self:
                case Failure(exception):
                    return Failure(exception)
                case Success(inner):
                    return await inner
                case _:
                    assert_never(outcome)

        return AsyncTry(run)

    def inspect(self, function: Callable[[T], None | Awaitable[None]]) -> AsyncTry[T]:
        """Inspect the AsyncTry's resolved value of T.

        Parameters
        ----------
        function: Callable[[T], None | Awaitable[None]]
            Sync or async inspection function called with the resolved value if Success.

        Returns
        -------
        async_try: AsyncTry[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(42).inspect(lambda x: print(f"Value is: {x}")))
        Value is: 42
        Success(42)
        """

        async def run() -> Try[T]:
            """Awaits self, calling `function` for its side effect only if Success."""
            match outcome := await self:
                case Success(value):
                    await _resolve(function(value))
                case Failure():
                    pass
                case _:
                    assert_never(outcome)
            return outcome

        return AsyncTry(run)

    def inspect_failure(
        self, function: Callable[[Exception], None | Awaitable[None]]
    ) -> AsyncTry[T]:
        """Inspect the AsyncTry's resolved Failure exception.

        Parameters
        ----------
        function: Callable[[Exception], None | Awaitable[None]]
            Sync or async inspection function called with the resolved exception if Failure.

        Returns
        -------
        async_try: AsyncTry[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).inspect_failure(lambda e: print(f"Exception: {e}")))
        Exception: boom
        Failure(ValueError('boom'))
        """

        async def run() -> Try[T]:
            """Awaits self, calling `function` for its side effect only if Failure."""
            match outcome := await self:
                case Failure(exc):
                    await _resolve(function(exc))
                case Success():
                    pass
                case _:
                    assert_never(outcome)
            return outcome

        return AsyncTry(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncTry.

        Examples
        --------
        >>> str(AsyncTry.from_success(10))  # doctest: +ELLIPSIS
        'AsyncTry(<function...>)'
        """
        return f"AsyncTry({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncTry (same as __str__).

        Examples
        --------
        >>> repr(AsyncTry.from_success(10))  # doctest: +ELLIPSIS
        'AsyncTry(<function...>)'
        """
        return str(self)


def map2(
    first: AsyncTry[A], second: AsyncTry[B], function: Callable[[A, B], C]
) -> AsyncTry[C]:
    """Combines the values of two AsyncTrys with a two-argument function once both are awaited.

    The first Failure wins: if `first` fails, `second` is not awaited.

    Examples
    --------
    >>> import asyncio
    >>> asyncio.run(map2(AsyncTry.from_success(1), AsyncTry.from_success(2), lambda a, b: a + b))
    Success(3)
    >>> asyncio.run(map2(AsyncTry.from_failure(ValueError("first")), AsyncTry.from_failure(ValueError("second")), lambda a, b: a + b))
    Failure(ValueError('first'))
    """
    return first.bind(lambda a: second.map(lambda b: function(a, b)))
