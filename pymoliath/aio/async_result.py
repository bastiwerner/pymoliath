"""
# AsyncResult

`AsyncResult` is the directly-awaitable counterpart of `pymoliath.result.Result` - see
`pymoliath.aio` for the general design shared by all `Async*` monads.

```python
import asyncio

asyncio.run(AsyncResult.from_ok(10).map(lambda x: x + 1))  # Ok(11)
asyncio.run(AsyncResult.from_err("error").map(lambda x: x + 1))  # Err(error), map is never called


async def fetch(x: int) -> int: ...


asyncio.run(AsyncResult.from_ok(10).map(fetch))  # async callback, auto-detected
asyncio.run(AsyncResult.from_ok(10).bind(lambda x: AsyncResult.from_ok(x + 1)))  # Ok(11)


async def fetch_ten() -> int:
    return 10


asyncio.run(AsyncResult.from_coroutine(fetch_ten))  # Ok(10)
```
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from typing import Any, Generic, Never, assert_never

from typing_extensions import TypeVar

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.result import Err, Ok, Result

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")

# Function-scoped TypeVars for the static constructors: the class-scoped T/E would be Unknown when
# called on the unspecialized class (`AsyncResult.from_ok(1)`).
V = TypeVar("V")
X = TypeVar("X")
# Like Ok's error type: `Never` unless the context (e.g. an annotation) asks for another one.
X_Never = TypeVar("X_Never", default=Never)
# Function-scoped TypeVars for `map2`.
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


class AsyncResult(Generic[T, E]):
    """Async Result Monad: a deferred computation which resolves to a Result[T, E] once awaited.

    Directly awaitable - nothing in a chain of map/map_err/bind/bind_err/... runs until the
    AsyncResult itself is awaited (`await an_async_result`), mirroring how Sequence
    (pymoliath/lazy.py) stays lazy until a terminal operation pulls from it, but for a single
    eventual Ok/Err value instead of an iterable.

    Callbacks passed to map/map_err/bind/bind_err/inspect/inspect_err may be plain sync functions
    or `async def` functions - whichever is returned is auto-detected at the point it's called
    (awaited only if it actually is an awaitable), so real async I/O can be mixed freely with plain
    transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Result[T, E]]]) -> None:
        """AsyncResult constructor which takes a zero-argument async callable resolving to a Result.

        Parameters
        ----------
        run: Callable[[], Awaitable[Result[T, E]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to a Result.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.

        Examples
        --------
        >>> import asyncio
        >>> async def run(): return Ok(10)
        >>> asyncio.run(AsyncResult(run))
        Ok(10)
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Result[T, E]]:
        """Runs the pipeline and resolves to the final Result[T, E].

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(10))
        Ok(10)
        """
        return self._run().__await__()

    @staticmethod
    def from_ok(value: V) -> AsyncResult[V, X_Never]:
        """Lifts a plain value into an already-Ok AsyncResult.

        Parameters
        ----------
        value: V
            Value to be wrapped as Ok once awaited.

        Returns
        -------
        async_result: AsyncResult[V, Never]
            Like a bare `Ok`, the error type is `Never` until the AsyncResult is annotated.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(10))
        Ok(10)
        """

        async def run() -> Result[V, X_Never]:
            """Resolves immediately to Ok(value)."""
            return Ok(value)

        return AsyncResult(run)

    @staticmethod
    # V is deliberately only in the return type: like a bare Err, the Ok type is left open to be
    # solved from context.
    def from_err(value: X) -> AsyncResult[V, X]:  # pyright: ignore[reportInvalidTypeVarUse]
        """Lifts a plain value into an already-Err AsyncResult.

        Parameters
        ----------
        value: X
            Value to be wrapped as Err once awaited.

        Returns
        -------
        async_result: AsyncResult[V, X]
            Like a bare `Err`, the Ok type is left open to be solved from context.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_err("error"))
        Err('error')
        """

        async def run() -> Result[V, X]:
            """Resolves immediately to Err(value)."""
            return Err(value)

        return AsyncResult(run)

    @staticmethod
    def from_result(result: Result[V, X]) -> AsyncResult[V, X]:
        """Lifts an existing sync Result (Ok or Err) into an AsyncResult.

        Parameters
        ----------
        result: Result[V, X]
            Result to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_result: AsyncResult[V, X]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_result(Ok(10)))
        Ok(10)
        >>> asyncio.run(AsyncResult.from_result(Err("error")))
        Err('error')
        """

        async def run() -> Result[V, X]:
            """Resolves immediately to `result`."""
            return result

        return AsyncResult(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[V]],
    ) -> AsyncResult[V, X_Never]:
        """Wraps a zero-argument async callable producing a raw value as an Ok once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[V]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncResult stays
            re-awaitable).

        Returns
        -------
        async_result: AsyncResult[V, Never]

        Examples
        --------
        >>> import asyncio
        >>> async def fetch_ten() -> int: return 10
        >>> asyncio.run(AsyncResult.from_coroutine(fetch_ten))
        Ok(10)
        """

        async def run() -> Result[V, X_Never]:
            """Awaits `coroutine_function` and wraps its result as Ok."""
            return Ok(await coroutine_function())

        return AsyncResult(run)

    def map(self, function: Callable[[T], U | Awaitable[U]]) -> AsyncResult[U, E]:
        """AsyncResult functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[T], U | Awaitable[U]]
            Sync or async function applied to the resolved value if Ok.

        Returns
        -------
        async_result: AsyncResult[U, E]
            Returns a new AsyncResult which resolves to Ok with the function result, or Err
            without calling `function`, if this AsyncResult resolves to Err.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(5).map(lambda x: x + 1))
        Ok(6)
        >>> asyncio.run(AsyncResult.from_err("error").map(lambda x: x + 1))
        Err('error')
        """

        async def run() -> Result[U, E]:
            """Awaits self, then applies `function`, short-circuiting on Err."""
            match outcome := await self:
                case Ok(value):
                    return Ok(await _resolve(function(value)))
                case Err(error):
                    return Err(error)
                case _:
                    assert_never(outcome)

        return AsyncResult(run)

    def map_err(self, function: Callable[[E], F | Awaitable[F]]) -> AsyncResult[T, F]:
        """AsyncResult functor interface for the Err channel.

        Parameters
        ----------
        function: Callable[[E], F | Awaitable[F]]
            Sync or async function applied to the resolved value if Err.

        Returns
        -------
        async_result: AsyncResult[T, F]
            Returns a new AsyncResult which resolves to Err with the function result, or Ok
            without calling `function`, if this AsyncResult resolves to Ok.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_err("error").map_err(str.upper))
        Err('ERROR')
        >>> asyncio.run(AsyncResult.from_ok(10).map_err(str.upper))
        Ok(10)
        """

        async def run() -> Result[T, F]:
            """Awaits self, then applies `function` to the Err value, short-circuiting on Ok."""
            match outcome := await self:
                case Ok(value):
                    return Ok(value)
                case Err(error):
                    return Err(await _resolve(function(error)))
                case _:
                    assert_never(outcome)

        return AsyncResult(run)

    def bind(
        self,
        function: Callable[
            [T],
            AsyncResult[U, E] | Result[U, E] | Awaitable[Result[U, E]],
        ],
    ) -> AsyncResult[U, E]:
        """AsyncResult bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[T], AsyncResult[U, E] | Result[U, E] | Awaitable[Result[U, E]]]
            Function applied to the resolved value if Ok, returning another AsyncResult, a plain
            Result, or an awaitable resolving to a Result - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_result: AsyncResult[U, E]
            Returns a new AsyncResult with the function result if Ok, otherwise Err without
            calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(5).bind(lambda x: AsyncResult.from_ok(x + 1)))
        Ok(6)
        >>> asyncio.run(AsyncResult.from_err("error").bind(lambda x: AsyncResult.from_ok(x + 1)))
        Err('error')
        """

        async def run() -> Result[U, E]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Err."""
            match outcome := await self:
                case Ok(value):
                    next_result = function(value)
                    if isinstance(next_result, AsyncResult):
                        return await next_result
                    return await _resolve(next_result)
                case Err(error):
                    return Err(error)
                case _:
                    assert_never(outcome)

        return AsyncResult(run)

    def bind_err(
        self,
        function: Callable[
            [E],
            AsyncResult[T, F] | Result[T, F] | Awaitable[Result[T, F]],
        ],
    ) -> AsyncResult[T, F]:
        """AsyncResult bind interface for the Err channel.

        Parameters
        ----------
        function: Callable[[E], AsyncResult[T, F] | Result[T, F] | Awaitable[Result[T, F]]]
            Function applied to the resolved value if Err, returning another AsyncResult, a plain
            Result, or an awaitable resolving to a Result - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_result: AsyncResult[T, F]
            Returns a new AsyncResult with the function result if Err, otherwise Ok without
            calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_err("error").bind_err(lambda e: AsyncResult.from_ok(0)))
        Ok(0)
        >>> asyncio.run(AsyncResult.from_ok(10).bind_err(lambda e: AsyncResult.from_ok(0)))
        Ok(10)
        """

        async def run() -> Result[T, F]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Ok."""
            match outcome := await self:
                case Ok(value):
                    return Ok(value)
                case Err(error):
                    next_result = function(error)
                    if isinstance(next_result, AsyncResult):
                        return await next_result
                    return await _resolve(next_result)
                case _:
                    assert_never(outcome)

        return AsyncResult(run)

    def apply(self, function: AsyncResult[Callable[[T], U], E]) -> AsyncResult[U, E]:
        """Applies the function wrapped in `function` to this AsyncResult's value (<*>).

        If both fail, the Err of `function` takes precedence, as in `Result.apply`. For functions of
        several arguments, use the module-level `map2`.

        Parameters
        ----------
        function: AsyncResult[Callable[[T], U], E]
            AsyncResult which contains a function of one argument.

        Returns
        -------
        async_result: AsyncResult[U, E]
            Returns the function applied to this AsyncResult's value.

        Examples
        --------
        >>> import asyncio
        >>> val = AsyncResult.from_ok(10)
        >>> func = AsyncResult.from_ok(lambda x: x * 2)
        >>> asyncio.run(val.apply(func))
        Ok(20)
        """
        return function.bind(lambda inner: self.map(inner))

    def apply2(
        self: AsyncResult[Callable[[U], V], E], value: AsyncResult[U, E]
    ) -> AsyncResult[V, E]:
        """Applies the function wrapped in this AsyncResult to the value wrapped in `value` (<*>).

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both fail, the
        Err of this (the function side) takes precedence. Curried functions of several arguments
        can be applied one argument at a time; `map2` takes them uncurried.

        Parameters
        ----------
        value: AsyncResult[U, E]
            AsyncResult which contains the argument.

        Returns
        -------
        async_value: AsyncResult[V, E]

        Examples
        --------
        >>> import asyncio
        >>> func = AsyncResult.from_ok(lambda y: 10 + y)
        >>> asyncio.run(func.apply2(AsyncResult.from_ok(5)))
        Ok(15)
        """
        return self.bind(lambda inner: value.map(inner))

    def and_(self, other: AsyncResult[U, E]) -> AsyncResult[U, E]:
        """Returns `other` if this AsyncResult resolves to Ok, otherwise Err.

        Parameters
        ----------
        other: AsyncResult[U, E]
            AsyncResult to be returned if this AsyncResult resolves to Ok.

        Returns
        -------
        async_result: AsyncResult[U, E]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(1).and_(AsyncResult.from_ok(2)))
        Ok(2)
        >>> asyncio.run(AsyncResult.from_err("error").and_(AsyncResult.from_ok(2)))
        Err('error')
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncResult[T, F]) -> AsyncResult[T, F]:
        """Returns this AsyncResult if it resolves to Ok, otherwise `other`.

        Parameters
        ----------
        other: AsyncResult[T, F]
            AsyncResult to be returned if this AsyncResult resolves to Err.

        Returns
        -------
        async_result: AsyncResult[T, F]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(1).or_(AsyncResult.from_ok(2)))
        Ok(1)
        >>> asyncio.run(AsyncResult.from_err("error").or_(AsyncResult.from_ok(2)))
        Ok(2)
        """

        async def run() -> Result[T, F]:
            """Awaits self, falling back to `other` if this AsyncResult resolves to Err."""
            match outcome := await self:
                case Ok(value):
                    return Ok(value)
                case Err():
                    return await other
                case _:
                    assert_never(outcome)

        return AsyncResult(run)

    def zip(self, other: AsyncResult[U, E]) -> AsyncResult[tuple[T, U], E]:
        """Combines this AsyncResult with another into an AsyncResult of a tuple, or Err if either is Err.

        Parameters
        ----------
        other: AsyncResult[U, E]
            AsyncResult to be zipped with this AsyncResult.

        Returns
        -------
        async_result: AsyncResult[tuple[T, U], E]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(1).zip(AsyncResult.from_ok(2)))
        Ok((1, 2))
        >>> asyncio.run(AsyncResult.from_err("error").zip(AsyncResult.from_ok(2)))
        Err('error')
        """

        async def run() -> Result[tuple[T, U], E]:
            """Awaits both self and `other`, combining their values if both are Ok."""
            match outcome := await self:
                case Ok(value):
                    return (await other).map(lambda other_value: (value, other_value))
                case Err(error):
                    return Err(error)
                case _:
                    assert_never(outcome)

        return AsyncResult(run)

    def flatten(
        self: AsyncResult[AsyncResult[U, E], E],
    ) -> AsyncResult[U, E]:
        """Flattens a nested AsyncResult by one level.

        Returns
        -------
        async_result: AsyncResult[U, E]
            Returns the nested AsyncResult's eventual result, or Err without awaiting it if this
            AsyncResult resolves to Err.

        Examples
        --------
        >>> import asyncio
        >>> nested = AsyncResult.from_ok(AsyncResult.from_ok(1))
        >>> asyncio.run(nested.flatten())
        Ok(1)
        """

        async def run() -> Result[U, E]:
            """Awaits self, then awaits the nested AsyncResult if Ok."""
            match outcome := await self:
                case Ok(nested):
                    return await nested
                case Err(error):
                    return Err(error)
                case _:
                    assert_never(outcome)

        return AsyncResult(run)

    def inspect(
        self, function: Callable[[T], None | Awaitable[None]]
    ) -> AsyncResult[T, E]:
        """Inspect the AsyncResult's resolved value of T.

        Parameters
        ----------
        function: Callable[[T], None | Awaitable[None]]
            Sync or async inspection function called with the resolved value if Ok.

        Returns
        -------
        async_result: AsyncResult[T, E]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(10).inspect(lambda x: print(f"Value: {x}")))
        Value: 10
        Ok(10)
        """

        async def run() -> Result[T, E]:
            """Awaits self, calling `function` for its side effect only if Ok."""
            outcome = await self
            if isinstance(outcome, Ok):
                await _resolve(function(outcome.value))
            return outcome

        return AsyncResult(run)

    def inspect_err(
        self, function: Callable[[E], None | Awaitable[None]]
    ) -> AsyncResult[T, E]:
        """Inspect the AsyncResult's resolved value of E.

        Parameters
        ----------
        function: Callable[[E], None | Awaitable[None]]
            Sync or async inspection function called with the resolved value if Err.

        Returns
        -------
        async_result: AsyncResult[T, E]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_err("error").inspect_err(lambda e: print(f"Error: {e}")))
        Error: error
        Err('error')
        """

        async def run() -> Result[T, E]:
            """Awaits self, calling `function` for its side effect only if Err."""
            outcome = await self
            if isinstance(outcome, Err):
                await _resolve(function(outcome.error))
            return outcome

        return AsyncResult(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncResult.

        Examples
        --------
        >>> str(AsyncResult.from_ok(10))  # doctest: +ELLIPSIS
        'AsyncResult(<function...>)'
        """
        return f"AsyncResult({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncResult (same as __str__).

        Examples
        --------
        >>> repr(AsyncResult.from_ok(10))  # doctest: +ELLIPSIS
        'AsyncResult(<function...>)'
        """
        return str(self)


def map2(
    first: AsyncResult[A, X], second: AsyncResult[B, X], function: Callable[[A, B], C]
) -> AsyncResult[C, X]:
    """Combines the values of two AsyncResults with a two-argument function once both are awaited.

    The first Err wins: if `first` fails, `second` is not awaited.

    Examples
    --------
    >>> import asyncio
    >>> asyncio.run(map2(AsyncResult.from_ok(1), AsyncResult.from_ok(2), lambda a, b: a + b))
    Ok(3)
    >>> asyncio.run(map2(AsyncResult.from_err("first"), AsyncResult.from_err("second"), lambda a, b: a + b))
    Err('first')
    """
    return first.bind(lambda a: second.map(lambda b: function(a, b)))
