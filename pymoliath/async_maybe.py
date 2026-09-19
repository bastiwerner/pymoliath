from __future__ import annotations

import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    Generic,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from pymoliath.maybe import Just, Maybe, Nothing
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


async def _resolve(value: Union[TypeResult, Awaitable[TypeResult]]) -> TypeResult:
    """Awaits `value` if it is awaitable, otherwise returns it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return cast(TypeResult, value)


class AsyncMaybe(Generic[TypeSource]):
    """Async Maybe Monad: a deferred computation which resolves to a Maybe[TypeSource] once awaited.

    Directly awaitable - nothing in a chain of map/bind/filter/... runs until the AsyncMaybe itself
    is awaited (`await an_async_maybe`), mirroring how Sequence (pymoliath/lazy.py) stays lazy until
    a terminal operation pulls from it, but for a single eventual value instead of an iterable.

    Callbacks passed to map/bind/filter/inspect may be plain sync functions or `async def` functions
    - whichever is returned is auto-detected at the point it's called (awaited only if it actually is
    an awaitable), so real async I/O can be mixed freely with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Maybe[TypeSource]]]) -> None:
        """AsyncMaybe constructor which takes a zero-argument async callable resolving to a Maybe.

        Parameters
        ----------
        run: Callable[[], Awaitable[Maybe[TypeSource]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to a Maybe.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Maybe[TypeSource]]:
        """Runs the pipeline and resolves to the final Maybe[TypeSource]."""
        return self._run().__await__()

    @staticmethod
    def from_value(value: TypeSource) -> AsyncMaybe[TypeSource]:
        """Lifts a plain value into an already-Just AsyncMaybe.

        Parameters
        ----------
        value: TypeSource
            Value to be wrapped as Just once awaited.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeSource]
        """

        async def run() -> Maybe[TypeSource]:
            """Resolves immediately to Just(value)."""
            return Just(value)

        return AsyncMaybe(run)

    @staticmethod
    def from_maybe(maybe: Maybe[TypeSource]) -> AsyncMaybe[TypeSource]:
        """Lifts an existing sync Maybe (Just or Nothing) into an AsyncMaybe.

        Parameters
        ----------
        maybe: Maybe[TypeSource]
            Maybe to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeSource]
        """

        async def run() -> Maybe[TypeSource]:
            """Resolves immediately to `maybe`."""
            return maybe

        return AsyncMaybe(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[TypeSource]],
    ) -> AsyncMaybe[TypeSource]:
        """Wraps a zero-argument async callable producing a raw value as a Just once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[TypeSource]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncMaybe stays
            re-awaitable).

        Returns
        -------
        async_maybe: AsyncMaybe[TypeSource]
        """

        async def run() -> Maybe[TypeSource]:
            """Awaits `coroutine_function` and wraps its result as Just."""
            return Just(await coroutine_function())

        return AsyncMaybe(run)

    def map(
        self, function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncMaybe[TypeResult]:
        """AsyncMaybe functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function applied to the resolved value if Just.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeResult]
            Returns a new AsyncMaybe which resolves to Just with the function result, or Nothing
            without calling `function`, if this AsyncMaybe resolves to Nothing.
        """

        async def run() -> Maybe[TypeResult]:
            """Awaits self, then applies `function`, short-circuiting on Nothing."""
            maybe = await self
            if maybe.is_nothing():
                return Nothing()
            return Just(await _resolve(function(maybe.unwrap())))

        return AsyncMaybe(run)

    def bind(
        self,
        function: Callable[
            [TypeSource],
            Union[AsyncMaybe[TypeResult], Maybe[TypeResult], Awaitable[Maybe[TypeResult]]],
        ],
    ) -> AsyncMaybe[TypeResult]:
        """AsyncMaybe bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], AsyncMaybe[TypeResult] | Maybe[TypeResult] | Awaitable[Maybe[TypeResult]]]
            Function applied to the resolved value if Just, returning another AsyncMaybe, a plain
            Maybe, or an awaitable resolving to a Maybe - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeResult]
            Returns a new AsyncMaybe with the function result if Just, otherwise Nothing without
            calling `function`.
        """

        async def run() -> Maybe[TypeResult]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Nothing."""
            maybe = await self
            if maybe.is_nothing():
                return Nothing()
            result = function(maybe.unwrap())
            if isinstance(result, AsyncMaybe):
                return await result
            return await _resolve(result)

        return AsyncMaybe(run)

    def apply(
        self, applicative: AsyncMaybe[Callable[..., TypeResult]]
    ) -> AsyncMaybe[TypeResult]:
        """AsyncMaybe applicative interface for AsyncMaybes containing a value (<*>).

        Parameters
        ----------
        applicative: AsyncMaybe[TypeApplicative] (TypeApplicative: any callable type)
            Applicative AsyncMaybe which contains a function and will be applied to the AsyncMaybe
            containing a value.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeResult]
            Applies an AsyncMaybe containing a value of type TypeSource to an AsyncMaybe containing
            a function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncMaybe[TypeResult]:
            """Maps the applicative's function, curried, over this AsyncMaybe's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: AsyncMaybe[Callable[..., TypeResult]],
        applicative_value: AsyncMaybe[Any],
    ) -> AsyncMaybe[TypeResult]:
        """AsyncMaybe applicative interface for AsyncMaybes containing a function (<*>).

        Parameters
        ----------
        applicative_value: AsyncMaybe[TypePure]
            AsyncMaybe value which will be applied to the AsyncMaybe containing a function.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeResult]
            Applies an AsyncMaybe containing a function to an AsyncMaybe of type TypePure (value or
            function).
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncMaybe[TypeResult]:
            """Maps the curried applicative function, held by this AsyncMaybe, over `applicative_value`."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def filter(
        self, filter_function: Callable[[TypeSource], Union[bool, Awaitable[bool]]]
    ) -> AsyncMaybe[TypeSource]:
        """Returns a Just if filter function is True and this AsyncMaybe resolves to Just, otherwise Nothing.

        Parameters
        ----------
        filter_function: Callable[[TypeSource], Union[bool, Awaitable[bool]]]
            Sync or async predicate applied to the resolved value if Just.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeSource]
            Returns an AsyncMaybe resolving to Just if this AsyncMaybe resolves to Just and
            `filter_function` returns True, otherwise Nothing.
        """

        async def run() -> Maybe[TypeSource]:
            """Awaits self, then keeps or discards the value based on `filter_function`."""
            maybe = await self
            if maybe.is_nothing():
                return maybe
            if await _resolve(filter_function(maybe.unwrap())):
                return maybe
            return Nothing()

        return AsyncMaybe(run)

    def and_(self, other: AsyncMaybe[TypeResult]) -> AsyncMaybe[TypeResult]:
        """Returns `other` if this AsyncMaybe resolves to Just, otherwise Nothing.

        Parameters
        ----------
        other: AsyncMaybe[TypeResult]
            AsyncMaybe to be returned if this AsyncMaybe resolves to Just.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeResult]
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncMaybe[TypeSource]) -> AsyncMaybe[TypeSource]:
        """Returns this AsyncMaybe if it resolves to Just, otherwise `other`.

        Parameters
        ----------
        other: AsyncMaybe[TypeSource]
            AsyncMaybe to be returned if this AsyncMaybe resolves to Nothing.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeSource]
        """

        async def run() -> Maybe[TypeSource]:
            """Awaits self, falling back to `other` if this AsyncMaybe resolves to Nothing."""
            maybe = await self
            if maybe.is_just():
                return maybe
            return await other

        return AsyncMaybe(run)

    def zip(self, other: AsyncMaybe[TypePure]) -> AsyncMaybe[Tuple[TypeSource, TypePure]]:
        """Combines this AsyncMaybe with another into an AsyncMaybe of a tuple, or Nothing if either is Nothing.

        Parameters
        ----------
        other: AsyncMaybe[TypePure]
            AsyncMaybe to be zipped with this AsyncMaybe.

        Returns
        -------
        async_maybe: AsyncMaybe[Tuple[TypeSource, TypePure]]
        """

        async def run() -> Maybe[Tuple[TypeSource, TypePure]]:
            """Awaits both self and `other`, combining their values if both are Just."""
            maybe = await self
            if maybe.is_nothing():
                return Nothing()
            other_maybe = await other
            return other_maybe.map(lambda o: (maybe.unwrap(), o))

        return AsyncMaybe(run)

    def flatten(self: AsyncMaybe[AsyncMaybe[TypeResult]]) -> AsyncMaybe[TypeResult]:
        """Flattens a nested AsyncMaybe by one level.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeResult]
            Returns the nested AsyncMaybe's eventual result, or Nothing without awaiting it if this
            AsyncMaybe resolves to Nothing.
        """

        async def run() -> Maybe[TypeResult]:
            """Awaits self, then awaits the nested AsyncMaybe if Just."""
            maybe = await self
            if maybe.is_nothing():
                return Nothing()
            return await maybe.unwrap()

        return AsyncMaybe(run)

    def inspect(
        self, function: Callable[[TypeSource], Union[None, Awaitable[None]]]
    ) -> AsyncMaybe[TypeSource]:
        """Inspect the AsyncMaybe's resolved value of TypeSource.

        Parameters
        ----------
        function: Callable[[TypeSource], Union[None, Awaitable[None]]]
            Sync or async inspection function called with the resolved value if Just.

        Returns
        -------
        async_maybe: AsyncMaybe[TypeSource]
        """

        async def run() -> Maybe[TypeSource]:
            """Awaits self, calling `function` for its side effect only if Just."""
            maybe = await self
            if maybe.is_just():
                await _resolve(function(maybe.unwrap()))
            return maybe

        return AsyncMaybe(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncMaybe."""
        return f"AsyncMaybe({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncMaybe (same as __str__)."""
        return str(self)
