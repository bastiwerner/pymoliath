"""
# AsyncMaybe

`AsyncMaybe` is the directly-awaitable counterpart of `pymoliath.maybe.Maybe` - see
`pymoliath.aio` for the general design shared by all `Async*` monads.

```python
import asyncio

asyncio.run(AsyncMaybe.from_value(10).map(lambda x: x + 1))  # Just(11)
asyncio.run(AsyncMaybe.from_maybe(Nothing()).map(lambda x: x + 1))  # Nothing(), map is never called


async def fetch(x: int) -> int: ...


asyncio.run(AsyncMaybe.from_value(10).map(fetch))  # async callback, auto-detected
asyncio.run(AsyncMaybe.from_value(10).bind(lambda x: AsyncMaybe.from_value(x + 1)))  # Just(11)
asyncio.run(AsyncMaybe.from_value(10).filter(lambda x: x > 5))  # Just(10)


async def fetch_ten() -> int:
    return 10


asyncio.run(AsyncMaybe.from_coroutine(fetch_ten))  # Just(10)
```
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from typing import Any, Generic, TypeVar, assert_never

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.maybe import Just, Maybe, Nothing

T = TypeVar("T")
U = TypeVar("U")

# Function-scoped TypeVar for the static constructors: the class-scoped T would be Unknown when
# called on the unspecialized class (`AsyncMaybe.from_value(1)`).
V = TypeVar("V")
# Function-scoped TypeVars for `map2`.
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


class AsyncMaybe(Generic[T]):
    """Async Maybe Monad: a deferred computation which resolves to a Maybe[T] once awaited.

    Directly awaitable - nothing in a chain of map/bind/filter/... runs until the AsyncMaybe itself
    is awaited (`await an_async_maybe`), mirroring how Sequence (pymoliath/lazy.py) stays lazy until
    a terminal operation pulls from it, but for a single eventual value instead of an iterable.

    Callbacks passed to map/bind/filter/inspect may be plain sync functions or `async def` functions
    - whichever is returned is auto-detected at the point it's called (awaited only if it actually is
    an awaitable), so real async I/O can be mixed freely with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Maybe[T]]]) -> None:
        """AsyncMaybe constructor which takes a zero-argument async callable resolving to a Maybe.

        Parameters
        ----------
        run: Callable[[], Awaitable[Maybe[T]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to a Maybe.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.

        Examples
        --------
        >>> import asyncio
        >>> async def run(): return Just(10)
        >>> asyncio.run(AsyncMaybe(run))
        Just(10)
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Maybe[T]]:
        """Runs the pipeline and resolves to the final Maybe[T].

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(10))
        Just(10)
        """
        return self._run().__await__()

    @staticmethod
    def from_value(value: V) -> AsyncMaybe[V]:
        """Lifts a plain value into an already-Just AsyncMaybe.

        Parameters
        ----------
        value: V
            Value to be wrapped as Just once awaited.

        Returns
        -------
        async_maybe: AsyncMaybe[V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(10))
        Just(10)
        """

        async def run() -> Maybe[V]:
            """Resolves immediately to Just(value)."""
            return Just(value)

        return AsyncMaybe(run)

    @staticmethod
    def from_maybe(maybe: Maybe[V]) -> AsyncMaybe[V]:
        """Lifts an existing sync Maybe (Just or Nothing) into an AsyncMaybe.

        Parameters
        ----------
        maybe: Maybe[V]
            Maybe to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_maybe: AsyncMaybe[V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_maybe(Just(10)))
        Just(10)
        >>> asyncio.run(AsyncMaybe.from_maybe(Nothing()))
        Nothing()
        """

        async def run() -> Maybe[V]:
            """Resolves immediately to `maybe`."""
            return maybe

        return AsyncMaybe(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[V]],
    ) -> AsyncMaybe[V]:
        """Wraps a zero-argument async callable producing a raw value as a Just once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[V]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncMaybe stays
            re-awaitable).

        Returns
        -------
        async_maybe: AsyncMaybe[V]

        Examples
        --------
        >>> import asyncio
        >>> async def fetch_ten() -> int: return 10
        >>> asyncio.run(AsyncMaybe.from_coroutine(fetch_ten))
        Just(10)
        """

        async def run() -> Maybe[V]:
            """Awaits `coroutine_function` and wraps its result as Just."""
            return Just(await coroutine_function())

        return AsyncMaybe(run)

    def map(self, function: Callable[[T], U | Awaitable[U]]) -> AsyncMaybe[U]:
        """AsyncMaybe functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[T], U | Awaitable[U]]
            Sync or async function applied to the resolved value if Just.

        Returns
        -------
        async_maybe: AsyncMaybe[U]
            Returns a new AsyncMaybe which resolves to Just with the function result, or Nothing
            without calling `function`, if this AsyncMaybe resolves to Nothing.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(5).map(lambda x: x + 1))
        Just(6)
        >>> asyncio.run(AsyncMaybe.from_maybe(Nothing()).map(lambda x: x + 1))
        Nothing()
        """

        async def run() -> Maybe[U]:
            """Awaits self, then applies `function`, short-circuiting on Nothing."""
            match maybe := await self:
                case Just(value):
                    return Just(await _resolve(function(value)))
                case Nothing():
                    return Nothing()
                case _:
                    assert_never(maybe)

        return AsyncMaybe(run)

    def bind(
        self,
        function: Callable[
            [T],
            AsyncMaybe[U] | Maybe[U] | Awaitable[Maybe[U]],
        ],
    ) -> AsyncMaybe[U]:
        """AsyncMaybe bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[T], AsyncMaybe[U] | Maybe[U] | Awaitable[Maybe[U]]]
            Function applied to the resolved value if Just, returning another AsyncMaybe, a plain
            Maybe, or an awaitable resolving to a Maybe - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_maybe: AsyncMaybe[U]
            Returns a new AsyncMaybe with the function result if Just, otherwise Nothing without
            calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(5).bind(lambda x: AsyncMaybe.from_value(x + 1)))
        Just(6)
        >>> asyncio.run(AsyncMaybe.from_maybe(Nothing()).bind(lambda x: AsyncMaybe.from_value(x + 1)))
        Nothing()
        """

        async def run() -> Maybe[U]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Nothing."""
            match maybe := await self:
                case Just(value):
                    result = function(value)
                    if isinstance(result, AsyncMaybe):
                        return await result
                    return await _resolve(result)
                case Nothing():
                    return Nothing()
                case _:
                    assert_never(maybe)

        return AsyncMaybe(run)

    def apply(self, function: AsyncMaybe[Callable[[T], U]]) -> AsyncMaybe[U]:
        """Applies the function wrapped in `function` to this AsyncMaybe's value (<*>).

        If both fail, Nothing takes precedence, as in `Maybe.apply`. For functions of
        several arguments, use the module-level `map2`.

        Parameters
        ----------
        function: AsyncMaybe[Callable[[T], U]]
            AsyncMaybe which contains a function of one argument.

        Returns
        -------
        async_maybe: AsyncMaybe[U]
            Returns the function applied to this AsyncMaybe's value.

        Examples
        --------
        >>> import asyncio
        >>> val = AsyncMaybe.from_value(10)
        >>> func = AsyncMaybe.from_value(lambda x: x * 2)
        >>> asyncio.run(val.apply(func))
        Just(20)
        """
        return function.bind(lambda inner: self.map(inner))

    def apply2(
        self: AsyncMaybe[Callable[[U], V]], value: AsyncMaybe[U]
    ) -> AsyncMaybe[V]:
        """Applies the function wrapped in this AsyncMaybe to the value wrapped in `value` (<*>).

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both fail, the
        Nothing of this (the function side) takes precedence. Curried functions of several arguments
        can be applied one argument at a time; `map2` takes them uncurried.

        Parameters
        ----------
        value: AsyncMaybe[U]
            AsyncMaybe which contains the argument.

        Returns
        -------
        async_value: AsyncMaybe[V]

        Examples
        --------
        >>> import asyncio
        >>> func = AsyncMaybe.from_value(lambda y: 10 + y)
        >>> asyncio.run(func.apply2(AsyncMaybe.from_value(5)))
        Just(15)
        """
        return self.bind(lambda inner: value.map(inner))

    def filter(
        self, filter_function: Callable[[T], bool | Awaitable[bool]]
    ) -> AsyncMaybe[T]:
        """Returns a Just if filter function is True and this AsyncMaybe resolves to Just, otherwise Nothing.

        Parameters
        ----------
        filter_function: Callable[[T], bool | Awaitable[bool]]
            Sync or async predicate applied to the resolved value if Just.

        Returns
        -------
        async_maybe: AsyncMaybe[T]
            Returns an AsyncMaybe resolving to Just if this AsyncMaybe resolves to Just and
            `filter_function` returns True, otherwise Nothing.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(10).filter(lambda x: x > 5))
        Just(10)
        >>> asyncio.run(AsyncMaybe.from_value(10).filter(lambda x: x < 5))
        Nothing()
        """

        async def run() -> Maybe[T]:
            """Awaits self, then keeps or discards the value based on `filter_function`."""
            match maybe := await self:
                case Just(value):
                    return (
                        maybe if await _resolve(filter_function(value)) else Nothing()
                    )
                case Nothing():
                    return maybe
                case _:
                    assert_never(maybe)

        return AsyncMaybe(run)

    def and_(self, other: AsyncMaybe[U]) -> AsyncMaybe[U]:
        """Returns `other` if this AsyncMaybe resolves to Just, otherwise Nothing.

        Parameters
        ----------
        other: AsyncMaybe[U]
            AsyncMaybe to be returned if this AsyncMaybe resolves to Just.

        Returns
        -------
        async_maybe: AsyncMaybe[U]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(1).and_(AsyncMaybe.from_value(2)))
        Just(2)
        >>> asyncio.run(AsyncMaybe.from_maybe(Nothing()).and_(AsyncMaybe.from_value(2)))
        Nothing()
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncMaybe[T]) -> AsyncMaybe[T]:
        """Returns this AsyncMaybe if it resolves to Just, otherwise `other`.

        Parameters
        ----------
        other: AsyncMaybe[T]
            AsyncMaybe to be returned if this AsyncMaybe resolves to Nothing.

        Returns
        -------
        async_maybe: AsyncMaybe[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(1).or_(AsyncMaybe.from_value(2)))
        Just(1)
        >>> asyncio.run(AsyncMaybe.from_maybe(Nothing()).or_(AsyncMaybe.from_value(2)))
        Just(2)
        """

        async def run() -> Maybe[T]:
            """Awaits self, falling back to `other` if this AsyncMaybe resolves to Nothing."""
            match maybe := await self:
                case Just():
                    return maybe
                case Nothing():
                    return await other
                case _:
                    assert_never(maybe)

        return AsyncMaybe(run)

    def zip(self, other: AsyncMaybe[U]) -> AsyncMaybe[tuple[T, U]]:
        """Combines this AsyncMaybe with another into an AsyncMaybe of a tuple, or Nothing if either is Nothing.

        Parameters
        ----------
        other: AsyncMaybe[U]
            AsyncMaybe to be zipped with this AsyncMaybe.

        Returns
        -------
        async_maybe: AsyncMaybe[tuple[T, U]]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(1).zip(AsyncMaybe.from_value(2)))
        Just((1, 2))
        >>> asyncio.run(AsyncMaybe.from_maybe(Nothing()).zip(AsyncMaybe.from_value(2)))
        Nothing()
        """

        async def run() -> Maybe[tuple[T, U]]:
            """Awaits both self and `other`, combining their values if both are Just."""
            match maybe := await self:
                case Just(value):
                    return (await other).map(lambda other_value: (value, other_value))
                case Nothing():
                    return Nothing()
                case _:
                    assert_never(maybe)

        return AsyncMaybe(run)

    def flatten(self: AsyncMaybe[AsyncMaybe[U]]) -> AsyncMaybe[U]:
        """Flattens a nested AsyncMaybe by one level.

        Returns
        -------
        async_maybe: AsyncMaybe[U]
            Returns the nested AsyncMaybe's eventual result, or Nothing without awaiting it if this
            AsyncMaybe resolves to Nothing.

        Examples
        --------
        >>> import asyncio
        >>> nested = AsyncMaybe.from_value(AsyncMaybe.from_value(1))
        >>> asyncio.run(nested.flatten())
        Just(1)
        """

        async def run() -> Maybe[U]:
            """Awaits self, then awaits the nested AsyncMaybe if Just."""
            match maybe := await self:
                case Just(nested):
                    return await nested
                case Nothing():
                    return Nothing()
                case _:
                    assert_never(maybe)

        return AsyncMaybe(run)

    def inspect(self, function: Callable[[T], None | Awaitable[None]]) -> AsyncMaybe[T]:
        """Inspect the AsyncMaybe's resolved value of T.

        Parameters
        ----------
        function: Callable[[T], None | Awaitable[None]]
            Sync or async inspection function called with the resolved value if Just.

        Returns
        -------
        async_maybe: AsyncMaybe[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncMaybe.from_value(42).inspect(lambda x: print(f"Value is: {x}")))
        Value is: 42
        Just(42)
        """

        async def run() -> Maybe[T]:
            """Awaits self, calling `function` for its side effect only if Just."""
            maybe = await self
            if isinstance(maybe, Just):
                await _resolve(function(maybe.value))
            return maybe

        return AsyncMaybe(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncMaybe.

        Examples
        --------
        >>> str(AsyncMaybe.from_value(10))  # doctest: +ELLIPSIS
        'AsyncMaybe(<function...>)'
        """
        return f"AsyncMaybe({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncMaybe (same as __str__).

        Examples
        --------
        >>> repr(AsyncMaybe.from_value(10))  # doctest: +ELLIPSIS
        'AsyncMaybe(<function...>)'
        """
        return str(self)


def map2(
    first: AsyncMaybe[A], second: AsyncMaybe[B], function: Callable[[A, B], C]
) -> AsyncMaybe[C]:
    """Combines the values of two AsyncMaybes with a two-argument function once both are awaited.

    The first Nothing wins: if `first` fails, `second` is not awaited.

    Examples
    --------
    >>> import asyncio
    >>> asyncio.run(map2(AsyncMaybe.from_value(1), AsyncMaybe.from_value(2), lambda a, b: a + b))
    Just(3)
    >>> asyncio.run(map2(AsyncMaybe.from_value(1), AsyncMaybe.from_maybe(Nothing()), lambda a, b: a + b))
    Nothing()
    """
    return first.bind(lambda a: second.map(lambda b: function(a, b)))
