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

from pymoliath._async import resolve as _resolve
from pymoliath.option import Nil, Option, Some
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class AsyncOption(Generic[TypeSource]):
    """Async Option Monad: a deferred computation which resolves to an Option[TypeSource] once awaited.

    Directly awaitable - nothing in a chain of map/bind/filter/... runs until the AsyncOption itself
    is awaited (`await an_async_option`), mirroring how Sequence (pymoliath/lazy.py) stays lazy until
    a terminal operation pulls from it, but for a single eventual value instead of an iterable.

    Callbacks passed to map/bind/filter/inspect may be plain sync functions or `async def` functions
    - whichever is returned is auto-detected at the point it's called (awaited only if it actually is
    an awaitable), so real async I/O can be mixed freely with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Option[TypeSource]]]) -> None:
        """AsyncOption constructor which takes a zero-argument async callable resolving to an Option.

        Parameters
        ----------
        run: Callable[[], Awaitable[Option[TypeSource]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to an Option.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Option[TypeSource]]:
        """Runs the pipeline and resolves to the final Option[TypeSource]."""
        return self._run().__await__()

    @staticmethod
    def from_value(value: TypeSource) -> AsyncOption[TypeSource]:
        """Lifts a plain value into an already-Some AsyncOption.

        Parameters
        ----------
        value: TypeSource
            Value to be wrapped as Some once awaited.

        Returns
        -------
        async_option: AsyncOption[TypeSource]
        """

        async def run() -> Option[TypeSource]:
            """Resolves immediately to Some(value)."""
            return Some(value)

        return AsyncOption(run)

    @staticmethod
    def from_option(option: Option[TypeSource]) -> AsyncOption[TypeSource]:
        """Lifts an existing sync Option (Some or Nil) into an AsyncOption.

        Parameters
        ----------
        option: Option[TypeSource]
            Option to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_option: AsyncOption[TypeSource]
        """

        async def run() -> Option[TypeSource]:
            """Resolves immediately to `option`."""
            return option

        return AsyncOption(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[TypeSource]],
    ) -> AsyncOption[TypeSource]:
        """Wraps a zero-argument async callable producing a raw value as a Some once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[TypeSource]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncOption stays
            re-awaitable).

        Returns
        -------
        async_option: AsyncOption[TypeSource]
        """

        async def run() -> Option[TypeSource]:
            """Awaits `coroutine_function` and wraps its result as Some."""
            return Some(await coroutine_function())

        return AsyncOption(run)

    def map(
        self, function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncOption[TypeResult]:
        """AsyncOption functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function applied to the resolved value if Some.

        Returns
        -------
        async_option: AsyncOption[TypeResult]
            Returns a new AsyncOption which resolves to Some with the function result, or Nil
            without calling `function`, if this AsyncOption resolves to Nil.
        """

        async def run() -> Option[TypeResult]:
            """Awaits self, then applies `function`, short-circuiting on Nil."""
            option = await self
            if option.is_nothing():
                return Nil()
            return Some(await _resolve(function(option.unwrap())))

        return AsyncOption(run)

    def bind(
        self,
        function: Callable[
            [TypeSource],
            Union[AsyncOption[TypeResult], Option[TypeResult], Awaitable[Option[TypeResult]]],
        ],
    ) -> AsyncOption[TypeResult]:
        """AsyncOption bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], AsyncOption[TypeResult] | Option[TypeResult] | Awaitable[Option[TypeResult]]]
            Function applied to the resolved value if Some, returning another AsyncOption, a plain
            Option, or an awaitable resolving to an Option - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_option: AsyncOption[TypeResult]
            Returns a new AsyncOption with the function result if Some, otherwise Nil without
            calling `function`.
        """

        async def run() -> Option[TypeResult]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Nil."""
            option = await self
            if option.is_nothing():
                return Nil()
            result = function(option.unwrap())
            if isinstance(result, AsyncOption):
                return await result
            return await _resolve(result)

        return AsyncOption(run)

    def apply(
        self, applicative: AsyncOption[Callable[..., TypeResult]]
    ) -> AsyncOption[TypeResult]:
        """AsyncOption applicative interface for AsyncOptions containing a value (<*>).

        Parameters
        ----------
        applicative: AsyncOption[TypeApplicative] (TypeApplicative: any callable type)
            Applicative AsyncOption which contains a function and will be applied to the AsyncOption
            containing a value.

        Returns
        -------
        async_option: AsyncOption[TypeResult]
            Applies an AsyncOption containing a value of type TypeSource to an AsyncOption containing
            a function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncOption[TypeResult]:
            """Maps the applicative's function, curried, over this AsyncOption's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: AsyncOption[Callable[..., TypeResult]],
        applicative_value: AsyncOption[Any],
    ) -> AsyncOption[TypeResult]:
        """AsyncOption applicative interface for AsyncOptions containing a function (<*>).

        Parameters
        ----------
        applicative_value: AsyncOption[TypePure]
            AsyncOption value which will be applied to the AsyncOption containing a function.

        Returns
        -------
        async_option: AsyncOption[TypeResult]
            Applies an AsyncOption containing a function to an AsyncOption of type TypePure (value or
            function).
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncOption[TypeResult]:
            """Maps the curried applicative function, held by this AsyncOption, over `applicative_value`."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def filter(
        self, filter_function: Callable[[TypeSource], Union[bool, Awaitable[bool]]]
    ) -> AsyncOption[TypeSource]:
        """Returns a Some if filter function is True and this AsyncOption resolves to Some, otherwise Nil.

        Parameters
        ----------
        filter_function: Callable[[TypeSource], Union[bool, Awaitable[bool]]]
            Sync or async predicate applied to the resolved value if Some.

        Returns
        -------
        async_option: AsyncOption[TypeSource]
            Returns an AsyncOption resolving to Some if this AsyncOption resolves to Some and
            `filter_function` returns True, otherwise Nil.
        """

        async def run() -> Option[TypeSource]:
            """Awaits self, then keeps or discards the value based on `filter_function`."""
            option = await self
            if option.is_nothing():
                return option
            if await _resolve(filter_function(option.unwrap())):
                return option
            return Nil()

        return AsyncOption(run)

    def and_(self, other: AsyncOption[TypeResult]) -> AsyncOption[TypeResult]:
        """Returns `other` if this AsyncOption resolves to Some, otherwise Nil.

        Parameters
        ----------
        other: AsyncOption[TypeResult]
            AsyncOption to be returned if this AsyncOption resolves to Some.

        Returns
        -------
        async_option: AsyncOption[TypeResult]
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncOption[TypeSource]) -> AsyncOption[TypeSource]:
        """Returns this AsyncOption if it resolves to Some, otherwise `other`.

        Parameters
        ----------
        other: AsyncOption[TypeSource]
            AsyncOption to be returned if this AsyncOption resolves to Nil.

        Returns
        -------
        async_option: AsyncOption[TypeSource]
        """

        async def run() -> Option[TypeSource]:
            """Awaits self, falling back to `other` if this AsyncOption resolves to Nil."""
            option = await self
            if option.is_some():
                return option
            return await other

        return AsyncOption(run)

    def zip(self, other: AsyncOption[TypePure]) -> AsyncOption[Tuple[TypeSource, TypePure]]:
        """Combines this AsyncOption with another into an AsyncOption of a tuple, or Nil if either is Nil.

        Parameters
        ----------
        other: AsyncOption[TypePure]
            AsyncOption to be zipped with this AsyncOption.

        Returns
        -------
        async_option: AsyncOption[Tuple[TypeSource, TypePure]]
        """

        async def run() -> Option[Tuple[TypeSource, TypePure]]:
            """Awaits both self and `other`, combining their values if both are Some."""
            option = await self
            if option.is_nothing():
                return Nil()
            other_option = await other
            return other_option.map(lambda o: (option.unwrap(), o))

        return AsyncOption(run)

    def flatten(self: AsyncOption[AsyncOption[TypeResult]]) -> AsyncOption[TypeResult]:
        """Flattens a nested AsyncOption by one level.

        Returns
        -------
        async_option: AsyncOption[TypeResult]
            Returns the nested AsyncOption's eventual result, or Nil without awaiting it if this
            AsyncOption resolves to Nil.
        """

        async def run() -> Option[TypeResult]:
            """Awaits self, then awaits the nested AsyncOption if Some."""
            option = await self
            if option.is_nothing():
                return Nil()
            return await option.unwrap()

        return AsyncOption(run)

    def inspect(
        self, function: Callable[[TypeSource], Union[None, Awaitable[None]]]
    ) -> AsyncOption[TypeSource]:
        """Inspect the AsyncOption's resolved value of TypeSource.

        Parameters
        ----------
        function: Callable[[TypeSource], Union[None, Awaitable[None]]]
            Sync or async inspection function called with the resolved value if Some.

        Returns
        -------
        async_option: AsyncOption[TypeSource]
        """

        async def run() -> Option[TypeSource]:
            """Awaits self, calling `function` for its side effect only if Some."""
            option = await self
            if option.is_some():
                await _resolve(function(option.unwrap()))
            return option

        return AsyncOption(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncOption."""
        return f"AsyncOption({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncOption (same as __str__)."""
        return str(self)
