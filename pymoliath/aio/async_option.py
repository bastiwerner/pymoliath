"""
# AsyncOption

`AsyncOption` is the directly-awaitable counterpart of `pymoliath.option.Option` - see
`pymoliath.aio` for the general design shared by all `Async*` monads.

```python
import asyncio

asyncio.run(AsyncOption.from_some(10).map(lambda x: x + 1).run())  # Some(11)
asyncio.run(AsyncOption.from_option(Nil()).map(lambda x: x + 1).run())  # Nil(), map is never called


async def fetch(x: int) -> int: ...


asyncio.run(AsyncOption.from_some(10).map(fetch).run())  # async callback, auto-detected
asyncio.run(AsyncOption.from_some(10).bind(lambda x: AsyncOption.from_some(x + 1)).run())  # Some(11)
asyncio.run(AsyncOption.from_some(10).filter(lambda x: x > 5).run())  # Some(10)


async def fetch_ten() -> int:
    return 10


asyncio.run(AsyncOption.from_coroutine(fetch_ten).run())  # Some(10)
```
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from typing import Any, Generic, TypeVar, assert_never

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.option import Nil, Option, Some

T = TypeVar("T")
U = TypeVar("U")

# Function-scoped TypeVar for the static constructors: the class-scoped T would be Unknown when
# called on the unspecialized class (`AsyncOption.from_some(1)`).
V = TypeVar("V")


class AsyncOption(Generic[T]):
    """Async Option Monad: a deferred computation which resolves to an Option[T] once awaited.

    Directly awaitable - nothing in a chain of map/bind/filter/... runs until the AsyncOption itself
    is awaited (`await an_async_option`), mirroring how Sequence (pymoliath/lazy.py) stays lazy until
    a terminal operation pulls from it, but for a single eventual value instead of an iterable.

    Callbacks passed to map/bind/filter/inspect may be plain sync functions or `async def` functions
    - whichever is returned is auto-detected at the point it's called (awaited only if it actually is
    an awaitable), so real async I/O can be mixed freely with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Option[T]]]) -> None:
        """AsyncOption constructor which takes a zero-argument async callable resolving to an Option.

        Parameters
        ----------
        run: Callable[[], Awaitable[Option[T]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to an Option.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.

        Examples
        --------
        >>> import asyncio
        >>> async def run(): return Some(10)
        >>> asyncio.run(AsyncOption(run).run())
        Some(10)
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Option[T]]:
        """Runs the pipeline and resolves to the final Option[T].

        Examples
        --------
        >>> import asyncio
        >>> async def main():
        ...     return await AsyncOption.from_some(10)
        >>> asyncio.run(main())
        Some(10)
        """
        return self._run().__await__()

    async def run(self) -> Option[T]:
        """Runs the pipeline and resolves to its result, as a coroutine.

        Equivalent to awaiting it directly. Use `run()` where a coroutine is required, e.g.
        `asyncio.run(value.run())` (before Python 3.14 `asyncio.run` only accepts coroutines).

        Returns
        -------
        result: Option[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(10).run())
        Some(10)
        """
        return await self

    @staticmethod
    def from_some(value: V) -> AsyncOption[V]:
        """Lifts a plain value into an already-Some AsyncOption.

        Parameters
        ----------
        value: V
            Value to be wrapped as Some once awaited.

        Returns
        -------
        async_option: AsyncOption[V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(10).run())
        Some(10)
        """

        async def run() -> Option[V]:
            """Resolves immediately to Some(value)."""
            return Some(value)

        return AsyncOption(run)

    @staticmethod
    # V is deliberately only in the return type: like a bare Nil, the value type is left open to
    # be solved from context.
    def from_nil() -> AsyncOption[V]:  # pyright: ignore[reportInvalidTypeVarUse]
        """Creates an already-Nil AsyncOption.

        Returns
        -------
        async_option: AsyncOption[V]
            Like a bare `Nil()`, the value type is left open to be solved from context; annotate
            the target (e.g. `empty: AsyncOption[int] = AsyncOption.from_nil()`) where there is none.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_nil().run())
        Nil()
        """

        async def run() -> Option[V]:
            """Resolves immediately to Nil()."""
            return Nil()

        return AsyncOption(run)

    @staticmethod
    def from_option(option: Option[V]) -> AsyncOption[V]:
        """Lifts an existing sync Option (Some or Nil) into an AsyncOption.

        Parameters
        ----------
        option: Option[V]
            Option to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_option: AsyncOption[V]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_option(Some(10)).run())
        Some(10)
        >>> asyncio.run(AsyncOption.from_option(Nil()).run())
        Nil()
        """

        async def run() -> Option[V]:
            """Resolves immediately to `option`."""
            return option

        return AsyncOption(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[V]],
    ) -> AsyncOption[V]:
        """Wraps a zero-argument async callable producing a raw value as a Some once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[V]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncOption stays
            re-awaitable).

        Returns
        -------
        async_option: AsyncOption[V]

        Examples
        --------
        >>> import asyncio
        >>> async def fetch_ten() -> int: return 10
        >>> asyncio.run(AsyncOption.from_coroutine(fetch_ten).run())
        Some(10)
        """

        async def run() -> Option[V]:
            """Awaits `coroutine_function` and wraps its result as Some."""
            return Some(await coroutine_function())

        return AsyncOption(run)

    def map(self, function: Callable[[T], U | Awaitable[U]]) -> AsyncOption[U]:
        """AsyncOption functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[T], U | Awaitable[U]]
            Sync or async function applied to the resolved value if Some.

        Returns
        -------
        async_option: AsyncOption[U]
            Returns a new AsyncOption which resolves to Some with the function result, or Nil
            without calling `function`, if this AsyncOption resolves to Nil.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(5).map(lambda x: x + 1).run())
        Some(6)
        >>> asyncio.run(AsyncOption.from_option(Nil()).map(lambda x: x + 1).run())
        Nil()
        """

        async def run() -> Option[U]:
            """Awaits self, then applies `function`, short-circuiting on Nil."""
            match option := await self:
                case Some(value):
                    return Some(await _resolve(function(value)))
                case Nil():
                    return option
                case _:
                    assert_never(option)

        return AsyncOption(run)

    def bind(
        self,
        function: Callable[
            [T],
            AsyncOption[U] | Option[U] | Awaitable[Option[U]],
        ],
    ) -> AsyncOption[U]:
        """AsyncOption bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[T], AsyncOption[U] | Option[U] | Awaitable[Option[U]]]
            Function applied to the resolved value if Some, returning another AsyncOption, a plain
            Option, or an awaitable resolving to an Option - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_option: AsyncOption[U]
            Returns a new AsyncOption with the function result if Some, otherwise Nil without
            calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(5).bind(lambda x: AsyncOption.from_some(x + 1)).run())
        Some(6)
        >>> asyncio.run(AsyncOption.from_option(Nil()).bind(lambda x: AsyncOption.from_some(x + 1)).run())
        Nil()
        """

        async def run() -> Option[U]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Nil."""
            match option := await self:
                case Some(value):
                    result = function(value)
                    if isinstance(result, AsyncOption):
                        return await result
                    return await _resolve(result)
                case Nil():
                    return option
                case _:
                    assert_never(option)

        return AsyncOption(run)

    def and_then(
        self,
        function: Callable[[T], AsyncOption[U] | Option[U] | Awaitable[Option[U]]],
    ) -> AsyncOption[U]:
        """Alias of `bind`, named like in Rust.

        Parameters
        ----------
        function: Callable[[T], AsyncOption[U] | Option[U] | Awaitable[Option[U]]]
            Function applied to the resolved value, as for `bind`.

        Returns
        -------
        async_option: AsyncOption[U]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(5).and_then(lambda x: AsyncOption.from_some(x + 1)).run())
        Some(6)
        """
        return self.bind(function)

    def or_else(
        self,
        function: Callable[[], AsyncOption[T] | Option[T] | Awaitable[Option[T]]],
    ) -> AsyncOption[T]:
        """Resolves to this AsyncOption if it is Some, otherwise to the result of `function`.

        Unlike `or_`, the fallback is only computed (and `function` only called) if this
        AsyncOption resolves to Nil.

        Parameters
        ----------
        function: Callable[[], AsyncOption[T] | Option[T] | Awaitable[Option[T]]]
            Zero-argument function computing the fallback: another AsyncOption, a plain Option, or an
            awaitable resolving to one.

        Returns
        -------
        async_option: AsyncOption[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_nil().or_else(lambda: AsyncOption.from_some(1)).run())
        Some(1)
        >>> asyncio.run(AsyncOption.from_some(2).or_else(lambda: AsyncOption.from_some(1)).run())
        Some(2)
        """

        async def run() -> Option[T]:
            """Awaits self, computing the fallback only if it resolves to Nil."""
            option = await self
            if isinstance(option, Nil):
                fallback = function()
                if isinstance(fallback, AsyncOption):
                    return await fallback
                return await _resolve(fallback)
            return option

        return AsyncOption(run)

    def apply(self, function: AsyncOption[Callable[[T], U]]) -> AsyncOption[U]:
        """Applies the function wrapped in `function` to this AsyncOption's value (<*>).

        If both fail, Nil takes precedence, as in `Option.apply`. For functions of
        several arguments, curry them and use `apply2`.

        Parameters
        ----------
        function: AsyncOption[Callable[[T], U]]
            AsyncOption which contains a function of one argument.

        Returns
        -------
        async_option: AsyncOption[U]
            Returns the function applied to this AsyncOption's value.

        Examples
        --------
        >>> import asyncio
        >>> val = AsyncOption.from_some(10)
        >>> func = AsyncOption.from_some(lambda x: x * 2)
        >>> asyncio.run(val.apply(func).run())
        Some(20)
        """
        return function.bind(lambda inner: self.map(inner))

    def apply2(
        self: AsyncOption[Callable[[U], V]], value: AsyncOption[U]
    ) -> AsyncOption[V]:
        """Applies the function wrapped in this AsyncOption to the value wrapped in `value` (<*>).

        The mirror image of `apply` (`func.apply2(val)` is `val.apply(func)`). If both fail, the
        Nil of this (the function side) takes precedence. Curried functions of several arguments
        can be applied one argument at a time.

        Parameters
        ----------
        value: AsyncOption[U]
            AsyncOption which contains the argument.

        Returns
        -------
        async_value: AsyncOption[V]

        Examples
        --------
        >>> import asyncio
        >>> func = AsyncOption.from_some(lambda y: 10 + y)
        >>> asyncio.run(func.apply2(AsyncOption.from_some(5)).run())
        Some(15)
        """
        return self.bind(lambda inner: value.map(inner))

    def filter(
        self, predicate: Callable[[T], bool | Awaitable[bool]]
    ) -> AsyncOption[T]:
        """Returns a Some if filter function is True and this AsyncOption resolves to Some, otherwise Nil.

        Parameters
        ----------
        predicate: Callable[[T], bool | Awaitable[bool]]
            Sync or async predicate applied to the resolved value if Some.

        Returns
        -------
        async_option: AsyncOption[T]
            Returns an AsyncOption resolving to Some if this AsyncOption resolves to Some and
            `predicate` returns True, otherwise Nil.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(10).filter(lambda x: x > 5).run())
        Some(10)
        >>> asyncio.run(AsyncOption.from_some(10).filter(lambda x: x < 5).run())
        Nil()
        """

        async def run() -> Option[T]:
            """Awaits self, then keeps or discards the value based on `predicate`."""
            match option := await self:
                case Some(value):
                    return option if await _resolve(predicate(value)) else Nil()
                case Nil():
                    return option
                case _:
                    assert_never(option)

        return AsyncOption(run)

    def and_(self, other: AsyncOption[U]) -> AsyncOption[U]:
        """Returns `other` if this AsyncOption resolves to Some, otherwise Nil.

        Parameters
        ----------
        other: AsyncOption[U]
            AsyncOption to be returned if this AsyncOption resolves to Some.

        Returns
        -------
        async_option: AsyncOption[U]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(1).and_(AsyncOption.from_some(2)).run())
        Some(2)
        >>> asyncio.run(AsyncOption.from_option(Nil()).and_(AsyncOption.from_some(2)).run())
        Nil()
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncOption[T]) -> AsyncOption[T]:
        """Returns this AsyncOption if it resolves to Some, otherwise `other`.

        Parameters
        ----------
        other: AsyncOption[T]
            AsyncOption to be returned if this AsyncOption resolves to Nil.

        Returns
        -------
        async_option: AsyncOption[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(1).or_(AsyncOption.from_some(2)).run())
        Some(1)
        >>> asyncio.run(AsyncOption.from_option(Nil()).or_(AsyncOption.from_some(2)).run())
        Some(2)
        """

        async def run() -> Option[T]:
            """Awaits self, falling back to `other` if this AsyncOption resolves to Nil."""
            match option := await self:
                case Some():
                    return option
                case Nil():
                    return await other
                case _:
                    assert_never(option)

        return AsyncOption(run)

    def zip(self, other: AsyncOption[U]) -> AsyncOption[tuple[T, U]]:
        """Combines this AsyncOption with another into an AsyncOption of a tuple, or Nil if either is Nil.

        Parameters
        ----------
        other: AsyncOption[U]
            AsyncOption to be zipped with this AsyncOption.

        Returns
        -------
        async_option: AsyncOption[tuple[T, U]]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(1).zip(AsyncOption.from_some(2)).run())
        Some((1, 2))
        >>> asyncio.run(AsyncOption.from_option(Nil()).zip(AsyncOption.from_some(2)).run())
        Nil()
        """

        async def run() -> Option[tuple[T, U]]:
            """Awaits both self and `other`, combining their values if both are Some."""
            match option := await self:
                case Some(value):
                    return (await other).map(lambda other_value: (value, other_value))
                case Nil():
                    return option
                case _:
                    assert_never(option)

        return AsyncOption(run)

    def flatten(self: AsyncOption[AsyncOption[U]]) -> AsyncOption[U]:
        """Flattens a nested AsyncOption by one level.

        Returns
        -------
        async_option: AsyncOption[U]
            Returns the nested AsyncOption's eventual result, or Nil without awaiting it if this
            AsyncOption resolves to Nil.

        Examples
        --------
        >>> import asyncio
        >>> nested = AsyncOption.from_some(AsyncOption.from_some(1))
        >>> asyncio.run(nested.flatten().run())
        Some(1)
        """

        async def run() -> Option[U]:
            """Awaits self, then awaits the nested AsyncOption if Some."""
            match option := await self:
                case Some(nested):
                    return await nested
                case Nil():
                    return option
                case _:
                    assert_never(option)

        return AsyncOption(run)

    def inspect(
        self, function: Callable[[T], None | Awaitable[None]]
    ) -> AsyncOption[T]:
        """Inspect the AsyncOption's resolved value of T.

        Parameters
        ----------
        function: Callable[[T], None | Awaitable[None]]
            Sync or async inspection function called with the resolved value if Some.

        Returns
        -------
        async_option: AsyncOption[T]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncOption.from_some(42).inspect(lambda x: print(f"Value is: {x}")).run())
        Value is: 42
        Some(42)
        """

        async def run() -> Option[T]:
            """Awaits self, calling `function` for its side effect only if Some."""
            option = await self
            if isinstance(option, Some):
                await _resolve(function(option.value))
            return option

        return AsyncOption(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncOption.

        Examples
        --------
        >>> str(AsyncOption.from_some(10))  # doctest: +ELLIPSIS
        'AsyncOption(<function...>)'
        """
        return f"AsyncOption({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncOption (same as __str__).

        Examples
        --------
        >>> repr(AsyncOption.from_some(10))  # doctest: +ELLIPSIS
        'AsyncOption(<function...>)'
        """
        return str(self)
