"""
# AsyncTry

`AsyncTry` is the directly-awaitable counterpart of `pymoliath.exception.Try` - see
`pymoliath.aio` for the general design shared by all `Async*` monads.

```python
import asyncio

asyncio.run(AsyncTry.from_success(10).map(lambda x: x + 1).run())  # Success(11)
asyncio.run(AsyncTry.from_success(10).map(lambda x: 1 / 0).run())  # Failure(...), exception caught


async def fetch(x: int) -> int: ...


asyncio.run(AsyncTry.from_success(10).map(fetch).run())  # async callback, auto-detected
asyncio.run(AsyncTry.from_success(10).bind(lambda x: AsyncTry.from_success(x + 1)).run())  # Success(11)


async def fetch_ten() -> int:
    return 10


asyncio.run(AsyncTry.from_coroutine(fetch_ten).run())  # Success(10)
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
    doesn't catch either - only the map/bind-family methods and `Success.safe()` do.

    Callbacks passed to map/bind/map_failure/bind_failure/inspect/inspect_failure may be plain
    sync functions or `async def` functions - whichever is returned is auto-detected at the point
    it's called (awaited only if it actually is an awaitable), so real async I/O can be mixed
    freely with plain transforms in the same chain.

    `from_failure` leaves the success type open, like a bare `Failure`: it is solved from context,
    but the two type checkers disagree on it without any (pyright: Unknown, mypy: Never). Annotate
    the target where there is no context, e.g.
    `failed: AsyncTry[int] = AsyncTry.from_failure(ValueError("e"))`.
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
        >>> asyncio.run(AsyncTry(run).run())
        Success(10)
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Try[T]]:
        """Runs the pipeline and resolves to the final Try[T].

        Examples
        --------
        >>> import asyncio
        >>> async def main():
        ...     return await AsyncTry.from_success(10)
        >>> asyncio.run(main())
        Success(10)
        """
        return self._run().__await__()

    async def run(self) -> Try[T]:
        """Runs the pipeline and resolves to its result, as a coroutine.

        Equivalent to awaiting it directly. Use `run()` where a coroutine is required, e.g.
        `asyncio.run(value.run())` (before Python 3.14 `asyncio.run` only accepts coroutines).

        Returns
        -------
        result: Try[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(10).run())
        Success(10)
        """
        return await self

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
        >>> asyncio.run(AsyncTry.from_success(10).run())
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
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).run())
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
        >>> asyncio.run(AsyncTry.from_try(Success(10)).run())
        Success(10)
        >>> asyncio.run(AsyncTry.from_try(Failure(ValueError("boom"))).run())
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
        >>> asyncio.run(AsyncTry.from_coroutine(fetch_ten).run())
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
        >>> asyncio.run(AsyncTry.from_success(5).map(lambda x: x + 1).run())
        Success(6)
        >>> def boom(x: int) -> int:
        ...     raise ValueError("boom")
        >>> asyncio.run(AsyncTry.from_success(5).map(boom).run())
        Failure(ValueError('boom'))
        """

        async def run() -> Try[U]:
            """Awaits self, then applies `function`, short-circuiting on Failure."""
            match outcome := await self:
                case Failure():
                    failed: Failure[Any] = outcome
                    return failed
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
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).map_failure(lambda e: TypeError(str(e))).run())
        Failure(TypeError('boom'))
        >>> asyncio.run(AsyncTry.from_success(10).map_failure(lambda e: TypeError(str(e))).run())
        Success(10)
        """

        async def run() -> Try[T]:
            """Awaits self, then applies `function` to the exception, short-circuiting on Success."""
            match outcome := await self:
                case Success():
                    return outcome
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
        >>> asyncio.run(AsyncTry.from_success(5).bind(lambda x: AsyncTry.from_success(x + 1)).run())
        Success(6)
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).bind(lambda x: AsyncTry.from_success(x + 1)).run())
        Failure(ValueError('boom'))
        """

        async def run() -> Try[U]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Failure."""
            match outcome := await self:
                case Failure():
                    failed: Failure[Any] = outcome
                    return failed
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

    def and_then(
        self,
        function: Callable[[T], AsyncTry[U] | Try[U] | Awaitable[Try[U]]],
    ) -> AsyncTry[U]:
        """Alias of `bind`, named like in Rust.

        Parameters
        ----------
        function: Callable[[T], AsyncTry[U] | Try[U] | Awaitable[Try[U]]]
            Function applied to the resolved value, as for `bind`.

        Returns
        -------
        async_try: AsyncTry[U]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(5).and_then(lambda x: AsyncTry.from_success(x + 1)).run())
        Success(6)
        """
        return self.bind(function)

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
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).bind_failure(lambda e: AsyncTry.from_success(0)).run())
        Success(0)
        >>> asyncio.run(AsyncTry.from_success(10).bind_failure(lambda e: AsyncTry.from_success(0)).run())
        Success(10)
        """

        async def run() -> Try[T]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Success."""
            match outcome := await self:
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
                case _:
                    assert_never(outcome)

        return AsyncTry(run)

    def or_else(
        self,
        function: Callable[[Exception], AsyncTry[T] | Try[T] | Awaitable[Try[T]]],
    ) -> AsyncTry[T]:
        """Alias of `bind_failure`, named like in Rust.

        Parameters
        ----------
        function: Callable[[Exception], AsyncTry[T] | Try[T] | Awaitable[Try[T]]]
            Function applied to the resolved error, as for `bind_failure`.

        Returns
        -------
        async_try: AsyncTry[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_failure(ValueError("e")).or_else(lambda e: AsyncTry.from_success(0)).run())
        Success(0)
        """
        return self.bind_failure(function)

    def apply(self, function: AsyncTry[Callable[[T], U]]) -> AsyncTry[U]:
        """Applies the function wrapped in `function` to this AsyncTry's value (<*>).

        If both fail, the Failure of `function` takes precedence, as in `Try.apply`. For functions of
        several arguments, curry them and use `apply2`.

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
        >>> asyncio.run(val.apply(func).run())
        Success(20)
        """
        return function.bind(lambda inner: self.map(inner))

    def apply2(self: AsyncTry[Callable[[U], V]], value: AsyncTry[U]) -> AsyncTry[V]:
        """Applies the function wrapped in this AsyncTry to the value wrapped in `value` (<*>).

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both fail, the
        Failure of this (the function side) takes precedence. Curried functions of several arguments
        can be applied one argument at a time.

        Parameters
        ----------
        value: AsyncTry[U]
            AsyncTry which contains the argument.

        Returns
        -------
        async_value: AsyncTry[V]

        Examples
        --------
        >>> import asyncio
        >>> func = AsyncTry.from_success(lambda y: 10 + y)
        >>> asyncio.run(func.apply2(AsyncTry.from_success(5)).run())
        Success(15)
        """
        return self.bind(lambda inner: value.map(inner))

    def filter(self, predicate: Callable[[T], bool | Awaitable[bool]]) -> AsyncTry[T]:
        """Keeps the Success value if the predicate holds for it, like Scala's `Try.filter`.

        A Success whose value fails the predicate resolves to
        `Failure(ValueError("predicate does not hold for <value>"))`, and a predicate that raises
        resolves to a Failure of that exception. A Failure stays unchanged.

        Parameters
        ----------
        predicate: Callable[[T], bool | Awaitable[bool]]
            Sync or async predicate applied to the resolved value if Success.

        Returns
        -------
        async_try: AsyncTry[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncTry.from_success(1).filter(lambda x: x > 0).run())
        Success(1)
        >>> asyncio.run(AsyncTry.from_success(-1).filter(lambda x: x > 0).run())
        Failure(ValueError('predicate does not hold for -1'))
        """

        async def run() -> Try[T]:
            """Awaits self, then keeps the Success value only if `predicate` holds."""
            outcome = await self
            if isinstance(outcome, Success):
                try:
                    holds = await _resolve(predicate(outcome.value))
                except Exception as e:
                    return Failure(e)
                if not holds:
                    message = f"predicate does not hold for {outcome.value!r}"
                    return Failure(ValueError(message))
            return outcome

        return AsyncTry(run)

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
        >>> asyncio.run(AsyncTry.from_success(1).and_(AsyncTry.from_success(2)).run())
        Success(2)
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).and_(AsyncTry.from_success(2)).run())
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
        >>> asyncio.run(AsyncTry.from_success(1).or_(AsyncTry.from_success(2)).run())
        Success(1)
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).or_(AsyncTry.from_success(2)).run())
        Success(2)
        """

        async def run() -> Try[T]:
            """Awaits self, falling back to `other` if this AsyncTry resolves to Failure."""
            match outcome := await self:
                case Success():
                    return outcome
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
        >>> asyncio.run(AsyncTry.from_success(1).zip(AsyncTry.from_success(2)).run())
        Success((1, 2))
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).zip(AsyncTry.from_success(2)).run())
        Failure(ValueError('boom'))
        """

        async def run() -> Try[tuple[T, U]]:
            """Awaits both self and `other`, combining their values if both are Success."""
            match outcome := await self:
                case Failure():
                    failed: Failure[Any] = outcome
                    return failed
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
        >>> asyncio.run(nested.flatten().run())
        Success(1)
        """

        async def run() -> Try[U]:
            """Awaits self, then awaits the nested AsyncTry if Success."""
            match outcome := await self:
                case Failure():
                    failed: Failure[Any] = outcome
                    return failed
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
        >>> asyncio.run(AsyncTry.from_success(42).inspect(lambda x: print(f"Value is: {x}")).run())
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
        >>> asyncio.run(AsyncTry.from_failure(ValueError("boom")).inspect_failure(lambda e: print(f"Exception: {e}")).run())
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
