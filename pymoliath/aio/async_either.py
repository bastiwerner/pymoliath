"""
# AsyncEither

`AsyncEither` is the directly-awaitable counterpart of `pymoliath.either.Either` - see
`pymoliath.aio` for the general design shared by all `Async*` monads.

```python
import asyncio

asyncio.run(AsyncEither.from_right(10).map(lambda x: x + 1).run())  # Right(11)
asyncio.run(AsyncEither.from_left("error").map(lambda x: x + 1).run())  # Left(error), map is never called


async def fetch(x: int) -> int: ...


asyncio.run(AsyncEither.from_right(10).map(fetch).run())  # async callback, auto-detected
asyncio.run(AsyncEither.from_right(10).bind(lambda x: AsyncEither.from_right(x + 1)).run())  # Right(11)


async def fetch_ten() -> int:
    return 10


asyncio.run(AsyncEither.from_coroutine(fetch_ten).run())  # Right(10)
```
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from typing import Any, Generic, Never, assert_never

from typing_extensions import TypeVar

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.either import Either, Left, Right

L = TypeVar("L")
R = TypeVar("R")
U = TypeVar("U")
F = TypeVar("F")

# Function-scoped TypeVars for the static constructors: the class-scoped L/R would be Unknown when
# called on the unspecialized class (`AsyncEither.from_right(1)`).
V = TypeVar("V")
W = TypeVar("W")
# Like Right's Left type: `Never` unless the context (e.g. an annotation) asks for another one.
W_Never = TypeVar("W_Never", default=Never)


class AsyncEither(Generic[L, R]):
    """Async Either Monad: a deferred computation which resolves to an Either[L, R] once awaited.

    Directly awaitable - nothing in a chain of map/map_left/bind/bind_left/... runs until the
    AsyncEither itself is awaited (`await an_async_either`), mirroring AsyncMaybe
    (pymoliath/async_maybe.py) but for the two-value (Left/Right) Either monad instead of
    Just/Nothing.

    Right-biased, matching the sync Either: map/bind/apply/and_/or_/zip/flatten/inspect
    all act on the Right channel and pass a Left through untouched; map_left/bind_left/inspect_left
    are the mirror image for the Left channel.

    Callbacks passed to map/map_left/bind/bind_left/inspect/inspect_left may be plain sync
    functions or `async def` functions - whichever is returned is auto-detected at the point it's
    called (awaited only if it actually is an awaitable), so real async I/O can be mixed freely
    with plain transforms in the same chain.

    `from_left` leaves the success type open, like a bare `Left`: it is solved from context, but
    the two type checkers disagree on it without any (pyright: Unknown, mypy: Never). Annotate the
    target where there is no context, e.g.
    `failed: AsyncEither[str, int] = AsyncEither.from_left("e")`.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Either[L, R]]]) -> None:
        """AsyncEither constructor which takes a zero-argument async callable resolving to an Either.

        Parameters
        ----------
        run: Callable[[], Awaitable[Either[L, R]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to an Either.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.

        Examples
        --------
        >>> import asyncio
        >>> async def run(): return Right(10)
        >>> asyncio.run(AsyncEither(run).run())
        Right(10)
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Either[L, R]]:
        """Runs the pipeline and resolves to the final Either[L, R].

        Examples
        --------
        >>> import asyncio
        >>> async def main():
        ...     return await AsyncEither.from_right(10)
        >>> asyncio.run(main())
        Right(10)
        """
        return self._run().__await__()

    async def run(self) -> Either[L, R]:
        """Runs the pipeline and resolves to its result, as a coroutine.

        Equivalent to awaiting it directly. Use `run()` where a coroutine is required, e.g.
        `asyncio.run(value.run())` (before Python 3.14 `asyncio.run` only accepts coroutines).

        Returns
        -------
        result: Either[L, R]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(10).run())
        Right(10)
        """
        return await self

    @staticmethod
    def from_right(value: V) -> AsyncEither[W_Never, V]:
        """Lifts a plain value into an already-Right AsyncEither.

        Parameters
        ----------
        value: V
            Value to be wrapped as Right once awaited.

        Returns
        -------
        async_either: AsyncEither[W_Never, V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(10).run())
        Right(10)
        """

        async def run() -> Either[W_Never, V]:
            """Resolves immediately to Right(value)."""
            return Right(value)

        return AsyncEither(run)

    @staticmethod
    # V is deliberately only in the return type: like a bare Left, the Right type is left open to be
    # solved from context.
    def from_left(value: W) -> AsyncEither[W, V]:  # pyright: ignore[reportInvalidTypeVarUse]
        """Lifts a plain value into an already-Left AsyncEither.

        Parameters
        ----------
        value: W
            Value to be wrapped as Left once awaited.

        Returns
        -------
        async_either: AsyncEither[W, V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_left("error").run())
        Left('error')
        """

        async def run() -> Either[W, V]:
            """Resolves immediately to Left(value)."""
            return Left(value)

        return AsyncEither(run)

    @staticmethod
    def from_either(
        either: Either[W, V],
    ) -> AsyncEither[W, V]:
        """Lifts an existing sync Either (Left or Right) into an AsyncEither.

        Parameters
        ----------
        either: Either[W, V]
            Either to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_either: AsyncEither[W, V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_either(Right(10)).run())
        Right(10)
        >>> asyncio.run(AsyncEither.from_either(Left("error")).run())
        Left('error')
        """

        async def run() -> Either[W, V]:
            """Resolves immediately to `either`."""
            return either

        return AsyncEither(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[V]],
    ) -> AsyncEither[W_Never, V]:
        """Wraps a zero-argument async callable producing a raw value as a Right once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[V]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncEither stays
            re-awaitable).

        Returns
        -------
        async_either: AsyncEither[W_Never, V]

        Examples
        --------
        >>> import asyncio
        >>> async def fetch_ten() -> int: return 10
        >>> asyncio.run(AsyncEither.from_coroutine(fetch_ten).run())
        Right(10)
        """

        async def run() -> Either[W_Never, V]:
            """Awaits `coroutine_function` and wraps its result as Right."""
            return Right(await coroutine_function())

        return AsyncEither(run)

    def map(self, function: Callable[[R], U | Awaitable[U]]) -> AsyncEither[L, U]:
        """AsyncEither functor interface (>=, map) over the Right channel.

        Parameters
        ----------
        function: Callable[[R], U | Awaitable[U]]
            Sync or async function applied to the resolved value if Right.

        Returns
        -------
        async_either: AsyncEither[L, U]
            Returns a new AsyncEither which resolves to Right with the function result, or the
            original Left untouched without calling `function`, if this AsyncEither resolves to
            Left.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(5).map(lambda x: x + 1).run())
        Right(6)
        >>> asyncio.run(AsyncEither.from_left("error").map(lambda x: x + 1).run())
        Left('error')
        """

        async def run() -> Either[L, U]:
            """Awaits self, then applies `function`, short-circuiting on Left."""
            match either := await self:
                case Left():
                    failed: Left[L, Any] = either
                    return failed
                case Right(value):
                    return Right(await _resolve(function(value)))
                case _:
                    assert_never(either)

        return AsyncEither(run)

    def map_left(self, function: Callable[[L], F | Awaitable[F]]) -> AsyncEither[F, R]:
        """AsyncEither functor interface (>=, map) over the Left channel.

        Parameters
        ----------
        function: Callable[[L], F | Awaitable[F]]
            Sync or async function applied to the resolved value if Left.

        Returns
        -------
        async_either: AsyncEither[F, R]
            Returns a new AsyncEither which resolves to Left with the function result, or the
            original Right untouched without calling `function`, if this AsyncEither resolves to
            Right.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_left("error").map_left(str.upper).run())
        Left('ERROR')
        >>> asyncio.run(AsyncEither.from_right(10).map_left(str.upper).run())
        Right(10)
        """

        async def run() -> Either[F, R]:
            """Awaits self, then applies `function`, short-circuiting on Right."""
            match either := await self:
                case Left(value):
                    return Left(await _resolve(function(value)))
                case Right():
                    succeeded: Right[Any, R] = either
                    return succeeded
                case _:
                    assert_never(either)

        return AsyncEither(run)

    def bind(
        self,
        function: Callable[
            [R],
            AsyncEither[L, U] | Either[L, U] | Awaitable[Either[L, U]],
        ],
    ) -> AsyncEither[L, U]:
        """AsyncEither bind interface (>>=, bind, flatMap) over the Right channel.

        Parameters
        ----------
        function: Callable[[R], AsyncEither[L, U] | Either[L, U] | Awaitable[Either[L, U]]]
            Function applied to the resolved value if Right, returning another AsyncEither, a
            plain Either, or an awaitable resolving to an Either - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_either: AsyncEither[L, U]
            Returns a new AsyncEither with the function result if Right, otherwise the original
            Left untouched without calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(5).bind(lambda x: AsyncEither.from_right(x + 1)).run())
        Right(6)
        >>> asyncio.run(AsyncEither.from_left("error").bind(lambda x: AsyncEither.from_right(x + 1)).run())
        Left('error')
        """

        async def run() -> Either[L, U]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Left."""
            match either := await self:
                case Left():
                    failed: Left[L, Any] = either
                    return failed
                case Right(value):
                    result = function(value)
                    if isinstance(result, AsyncEither):
                        return await result
                    return await _resolve(result)
                case _:
                    assert_never(either)

        return AsyncEither(run)

    def and_then(
        self,
        function: Callable[
            [R], AsyncEither[L, U] | Either[L, U] | Awaitable[Either[L, U]]
        ],
    ) -> AsyncEither[L, U]:
        """Alias of `bind`, named like in Rust.

        Parameters
        ----------
        function: Callable[[R], AsyncEither[L, U] | Either[L, U] | Awaitable[Either[L, U]]]
            Function applied to the resolved value, as for `bind`.

        Returns
        -------
        async_either: AsyncEither[L, U]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(5).and_then(lambda x: AsyncEither.from_right(x + 1)).run())
        Right(6)
        """
        return self.bind(function)

    def bind_left(
        self,
        function: Callable[
            [L],
            AsyncEither[F, R] | Either[F, R] | Awaitable[Either[F, R]],
        ],
    ) -> AsyncEither[F, R]:
        """AsyncEither bind interface (>>=, bind, flatMap) over the Left channel.

        Parameters
        ----------
        function: Callable[[L], AsyncEither[F, R] | Either[F, R] | Awaitable[Either[F, R]]]
            Function applied to the resolved value if Left, returning another AsyncEither, a
            plain Either, or an awaitable resolving to an Either - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_either: AsyncEither[F, R]
            Returns a new AsyncEither with the function result if Left, otherwise the original
            Right untouched without calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_left("error").bind_left(lambda e: AsyncEither.from_right(0)).run())
        Right(0)
        >>> asyncio.run(AsyncEither.from_right(10).bind_left(lambda e: AsyncEither.from_right(0)).run())
        Right(10)
        """

        async def run() -> Either[F, R]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Right."""
            match either := await self:
                case Left(value):
                    result = function(value)
                    if isinstance(result, AsyncEither):
                        return await result
                    return await _resolve(result)
                case Right():
                    succeeded: Right[Any, R] = either
                    return succeeded
                case _:
                    assert_never(either)

        return AsyncEither(run)

    def or_else(
        self,
        function: Callable[
            [L], AsyncEither[F, R] | Either[F, R] | Awaitable[Either[F, R]]
        ],
    ) -> AsyncEither[F, R]:
        """Alias of `bind_left`, named like in Rust.

        Parameters
        ----------
        function: Callable[[L], AsyncEither[F, R] | Either[F, R] | Awaitable[Either[F, R]]]
            Function applied to the resolved error, as for `bind_left`.

        Returns
        -------
        async_either: AsyncEither[F, R]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_left("e").or_else(lambda e: AsyncEither.from_right(0)).run())
        Right(0)
        """
        return self.bind_left(function)

    def apply(self, function: AsyncEither[L, Callable[[R], U]]) -> AsyncEither[L, U]:
        """Applies the function wrapped in `function` to this AsyncEither's value (<*>).

        If both fail, the Left of `function` takes precedence, as in `Either.apply`. For functions of
        several arguments, curry them and use `apply2`.

        Parameters
        ----------
        function: AsyncEither[L, Callable[[R], U]]
            AsyncEither which contains a function of one argument.

        Returns
        -------
        async_either: AsyncEither[L, U]
            Returns the function applied to this AsyncEither's value.

        Examples
        --------
        >>> import asyncio
        >>> val = AsyncEither.from_right(10)
        >>> func = AsyncEither.from_right(lambda x: x * 2)
        >>> asyncio.run(val.apply(func).run())
        Right(20)
        """
        return function.bind(lambda inner: self.map(inner))

    def apply2(
        self: AsyncEither[L, Callable[[U], V]], value: AsyncEither[L, U]
    ) -> AsyncEither[L, V]:
        """Applies the function wrapped in this AsyncEither to the value wrapped in `value` (<*>).

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both fail, the
        Left of this (the function side) takes precedence. Curried functions of several arguments
        can be applied one argument at a time.

        Parameters
        ----------
        value: AsyncEither[L, U]
            AsyncEither which contains the argument.

        Returns
        -------
        async_value: AsyncEither[L, V]

        Examples
        --------
        >>> import asyncio
        >>> func = AsyncEither.from_right(lambda y: 10 + y)
        >>> asyncio.run(func.apply2(AsyncEither.from_right(5)).run())
        Right(15)
        """
        return self.bind(lambda inner: value.map(inner))

    def filter(
        self, predicate: Callable[[R], bool | Awaitable[bool]], left_value: L
    ) -> AsyncEither[L, R]:
        """Keeps the Right value if the predicate holds for it, otherwise resolves to
        Left(left_value).

        Parameters
        ----------
        predicate: Callable[[R], bool | Awaitable[bool]]
            Sync or async predicate applied to the resolved value if Right.
        left_value: L
            Left value used if the predicate does not hold.

        Returns
        -------
        async_either: AsyncEither[L, R]
            Resolves to the Right, to Left(left_value) if the predicate fails, or to the original
            Left without calling `predicate`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(-1).filter(lambda x: x > 0, "negative").run())
        Left('negative')
        >>> asyncio.run(AsyncEither.from_right(1).filter(lambda x: x > 0, "negative").run())
        Right(1)
        """

        async def run() -> Either[L, R]:
            """Awaits self, then keeps the Right value only if `predicate` holds."""
            outcome = await self
            if isinstance(outcome, Right) and not await _resolve(
                predicate(outcome.value)
            ):
                return Left(left_value)
            return outcome

        return AsyncEither(run)

    def and_(self, other: AsyncEither[L, U]) -> AsyncEither[L, U]:
        """Returns `other` if this AsyncEither resolves to Right, otherwise the original Left.

        Parameters
        ----------
        other: AsyncEither[L, U]
            AsyncEither to be returned if this AsyncEither resolves to Right.

        Returns
        -------
        async_either: AsyncEither[L, U]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(1).and_(AsyncEither.from_right(2)).run())
        Right(2)
        >>> asyncio.run(AsyncEither.from_left("error").and_(AsyncEither.from_right(2)).run())
        Left('error')
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncEither[F, R]) -> AsyncEither[F, R]:
        """Returns this AsyncEither if it resolves to Right, otherwise `other`.

        Parameters
        ----------
        other: AsyncEither[F, R]
            AsyncEither to be returned if this AsyncEither resolves to Left.

        Returns
        -------
        async_either: AsyncEither[F, R]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(1).or_(AsyncEither.from_right(2)).run())
        Right(1)
        >>> asyncio.run(AsyncEither.from_left("error").or_(AsyncEither.from_right(2)).run())
        Right(2)
        """

        async def run() -> Either[F, R]:
            """Awaits self, falling back to `other` if this AsyncEither resolves to Left."""
            match either := await self:
                case Left():
                    return await other
                case Right():
                    succeeded: Right[Any, R] = either
                    return succeeded
                case _:
                    assert_never(either)

        return AsyncEither(run)

    def zip(self, other: AsyncEither[L, U]) -> AsyncEither[L, tuple[R, U]]:
        """Combines this AsyncEither with another into an AsyncEither of a tuple, or Left if either is Left.

        Parameters
        ----------
        other: AsyncEither[L, U]
            AsyncEither to be zipped with this AsyncEither.

        Returns
        -------
        async_either: AsyncEither[L, tuple[R, U]]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(1).zip(AsyncEither.from_right(2)).run())
        Right((1, 2))
        >>> asyncio.run(AsyncEither.from_left("error").zip(AsyncEither.from_right(2)).run())
        Left('error')
        """

        async def run() -> Either[L, tuple[R, U]]:
            """Awaits both self and `other`, combining their values if both are Right."""
            match either := await self:
                case Left():
                    failed: Left[L, Any] = either
                    return failed
                case Right(value):
                    return (await other).map(lambda other_value: (value, other_value))
                case _:
                    assert_never(either)

        return AsyncEither(run)

    def flatten(
        self: AsyncEither[L, AsyncEither[L, U]],
    ) -> AsyncEither[L, U]:
        """Flattens a nested AsyncEither by one level.

        Returns
        -------
        async_either: AsyncEither[L, U]
            Returns the nested AsyncEither's eventual result, or the original Left without
            awaiting it if this AsyncEither resolves to Left.

        Examples
        --------
        >>> import asyncio
        >>> nested = AsyncEither.from_right(AsyncEither.from_right(1))
        >>> asyncio.run(nested.flatten().run())
        Right(1)
        """

        async def run() -> Either[L, U]:
            """Awaits self, then awaits the nested AsyncEither if Right."""
            match either := await self:
                case Left():
                    failed: Left[L, Any] = either
                    return failed
                case Right(nested):
                    return await nested
                case _:
                    assert_never(either)

        return AsyncEither(run)

    def inspect(
        self, function: Callable[[R], None | Awaitable[None]]
    ) -> AsyncEither[L, R]:
        """Inspect the AsyncEither's resolved value of R.

        Parameters
        ----------
        function: Callable[[R], None | Awaitable[None]]
            Sync or async inspection function called with the resolved value if Right.

        Returns
        -------
        async_either: AsyncEither[L, R]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_right(10).inspect(lambda x: print(f"Value: {x}")).run())
        Value: 10
        Right(10)
        """

        async def run() -> Either[L, R]:
            """Awaits self, calling `function` for its side effect only if Right."""
            either = await self
            if isinstance(either, Right):
                await _resolve(function(either.value))
            return either

        return AsyncEither(run)

    def inspect_left(
        self, function: Callable[[L], None | Awaitable[None]]
    ) -> AsyncEither[L, R]:
        """Inspect the AsyncEither's resolved value of L.

        Parameters
        ----------
        function: Callable[[L], None | Awaitable[None]]
            Sync or async inspection function called with the resolved value if Left.

        Returns
        -------
        async_either: AsyncEither[L, R]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncEither.from_left("error").inspect_left(lambda e: print(f"Left: {e}")).run())
        Left: error
        Left('error')
        """

        async def run() -> Either[L, R]:
            """Awaits self, calling `function` for its side effect only if Left."""
            either = await self
            if isinstance(either, Left):
                await _resolve(function(either.value))
            return either

        return AsyncEither(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncEither.

        Examples
        --------
        >>> str(AsyncEither.from_right(10))  # doctest: +ELLIPSIS
        'AsyncEither(<function...>)'
        """
        return f"AsyncEither({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncEither (same as __str__).

        Examples
        --------
        >>> repr(AsyncEither.from_right(10))  # doctest: +ELLIPSIS
        'AsyncEither(<function...>)'
        """
        return str(self)
