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
from pymoliath.either import Either, Left, Right
from pymoliath.util import curry

TypeLeft = TypeVar("TypeLeft")
TypeRight = TypeVar("TypeRight")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class AsyncEither(Generic[TypeLeft, TypeRight]):
    """Async Either Monad: a deferred computation which resolves to an Either[TypeLeft, TypeRight] once awaited.

    Directly awaitable - nothing in a chain of map/map_left/bind/bind_left/... runs until the
    AsyncEither itself is awaited (`await an_async_either`), mirroring AsyncMaybe
    (pymoliath/async_maybe.py) but for the two-value (Left/Right) Either monad instead of
    Just/Nothing.

    Right-biased, matching the sync Either: map/bind/apply/apply2/and_/or_/zip/flatten/inspect
    all act on the Right channel and pass a Left through untouched; map_left/bind_left/inspect_left
    are the mirror image for the Left channel.

    Callbacks passed to map/map_left/bind/bind_left/inspect/inspect_left may be plain sync
    functions or `async def` functions - whichever is returned is auto-detected at the point it's
    called (awaited only if it actually is an awaitable), so real async I/O can be mixed freely
    with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Either[TypeLeft, TypeRight]]]) -> None:
        """AsyncEither constructor which takes a zero-argument async callable resolving to an Either.

        Parameters
        ----------
        run: Callable[[], Awaitable[Either[TypeLeft, TypeRight]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to an Either.
            To stay re-awaitable, `run` must produce a *new* awaitable every call rather than
            handing back an already-created (and possibly already-consumed) coroutine object -
            the same caveat Sequence's docstring makes about raw generators/iterators.
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Either[TypeLeft, TypeRight]]:
        """Runs the pipeline and resolves to the final Either[TypeLeft, TypeRight]."""
        return self._run().__await__()

    @staticmethod
    def from_right(value: TypeRight) -> AsyncEither[Any, TypeRight]:
        """Lifts a plain value into an already-Right AsyncEither.

        Parameters
        ----------
        value: TypeRight
            Value to be wrapped as Right once awaited.

        Returns
        -------
        async_either: AsyncEither[Any, TypeRight]
        """

        async def run() -> Either[Any, TypeRight]:
            """Resolves immediately to Right(value)."""
            return Right(value)

        return AsyncEither(run)

    @staticmethod
    def from_left(value: TypeLeft) -> AsyncEither[TypeLeft, Any]:
        """Lifts a plain value into an already-Left AsyncEither.

        Parameters
        ----------
        value: TypeLeft
            Value to be wrapped as Left once awaited.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, Any]
        """

        async def run() -> Either[TypeLeft, Any]:
            """Resolves immediately to Left(value)."""
            return Left(value)

        return AsyncEither(run)

    @staticmethod
    def from_either(either: Either[TypeLeft, TypeRight]) -> AsyncEither[TypeLeft, TypeRight]:
        """Lifts an existing sync Either (Left or Right) into an AsyncEither.

        Parameters
        ----------
        either: Either[TypeLeft, TypeRight]
            Either to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeRight]
        """

        async def run() -> Either[TypeLeft, TypeRight]:
            """Resolves immediately to `either`."""
            return either

        return AsyncEither(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[TypeRight]],
    ) -> AsyncEither[Any, TypeRight]:
        """Wraps a zero-argument async callable producing a raw value as a Right once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[TypeRight]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncEither stays
            re-awaitable).

        Returns
        -------
        async_either: AsyncEither[Any, TypeRight]
        """

        async def run() -> Either[Any, TypeRight]:
            """Awaits `coroutine_function` and wraps its result as Right."""
            return Right(await coroutine_function())

        return AsyncEither(run)

    def map(
        self, function: Callable[[TypeRight], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncEither[TypeLeft, TypeResult]:
        """AsyncEither functor interface (>=, map) over the Right channel.

        Parameters
        ----------
        function: Callable[[TypeRight], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function applied to the resolved value if Right.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeResult]
            Returns a new AsyncEither which resolves to Right with the function result, or the
            original Left untouched without calling `function`, if this AsyncEither resolves to
            Left.
        """

        async def run() -> Either[TypeLeft, TypeResult]:
            """Awaits self, then applies `function`, short-circuiting on Left."""
            either = await self
            if isinstance(either, Left):
                return either
            return Right(await _resolve(function(either.unwrap())))

        return AsyncEither(run)

    def map_left(
        self, function: Callable[[TypeLeft], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncEither[TypeResult, TypeRight]:
        """AsyncEither functor interface (>=, map) over the Left channel.

        Parameters
        ----------
        function: Callable[[TypeLeft], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function applied to the resolved value if Left.

        Returns
        -------
        async_either: AsyncEither[TypeResult, TypeRight]
            Returns a new AsyncEither which resolves to Left with the function result, or the
            original Right untouched without calling `function`, if this AsyncEither resolves to
            Right.
        """

        async def run() -> Either[TypeResult, TypeRight]:
            """Awaits self, then applies `function`, short-circuiting on Right."""
            either = await self
            match either:
                case Left(value):
                    return Left(await _resolve(function(value)))
                case _:
                    return either

        return AsyncEither(run)

    def bind(
        self,
        function: Callable[
            [TypeRight],
            Union[
                AsyncEither[TypeLeft, TypeResult],
                Either[TypeLeft, TypeResult],
                Awaitable[Either[TypeLeft, TypeResult]],
            ],
        ],
    ) -> AsyncEither[TypeLeft, TypeResult]:
        """AsyncEither bind interface (>>=, bind, flatMap) over the Right channel.

        Parameters
        ----------
        function: Callable[[TypeRight], AsyncEither[TypeLeft, TypeResult] | Either[TypeLeft, TypeResult] | Awaitable[Either[TypeLeft, TypeResult]]]
            Function applied to the resolved value if Right, returning another AsyncEither, a
            plain Either, or an awaitable resolving to an Either - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeResult]
            Returns a new AsyncEither with the function result if Right, otherwise the original
            Left untouched without calling `function`.
        """

        async def run() -> Either[TypeLeft, TypeResult]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Left."""
            either = await self
            if isinstance(either, Left):
                return either
            result = function(either.unwrap())
            if isinstance(result, AsyncEither):
                return await result
            return await _resolve(result)

        return AsyncEither(run)

    def bind_left(
        self,
        function: Callable[
            [TypeLeft],
            Union[
                AsyncEither[TypeResult, TypeRight],
                Either[TypeResult, TypeRight],
                Awaitable[Either[TypeResult, TypeRight]],
            ],
        ],
    ) -> AsyncEither[TypeResult, TypeRight]:
        """AsyncEither bind interface (>>=, bind, flatMap) over the Left channel.

        Parameters
        ----------
        function: Callable[[TypeLeft], AsyncEither[TypeResult, TypeRight] | Either[TypeResult, TypeRight] | Awaitable[Either[TypeResult, TypeRight]]]
            Function applied to the resolved value if Left, returning another AsyncEither, a
            plain Either, or an awaitable resolving to an Either - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_either: AsyncEither[TypeResult, TypeRight]
            Returns a new AsyncEither with the function result if Left, otherwise the original
            Right untouched without calling `function`.
        """

        async def run() -> Either[TypeResult, TypeRight]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Right."""
            either = await self
            match either:
                case Left(value):
                    result = function(value)
                    if isinstance(result, AsyncEither):
                        return await result
                    return await _resolve(result)
                case _:
                    return either

        return AsyncEither(run)

    def apply(
        self, applicative: AsyncEither[TypeLeft, Callable[..., TypeResult]]
    ) -> AsyncEither[TypeLeft, TypeResult]:
        """AsyncEither applicative interface for AsyncEithers containing a value (<*>).

        Parameters
        ----------
        applicative: AsyncEither[TypeLeft, TypeApplicative] (TypeApplicative: any callable type)
            Applicative AsyncEither which contains a function and will be applied to the
            AsyncEither containing a value.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeResult]
            Applies an AsyncEither containing a value of type TypeRight to an AsyncEither
            containing a function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncEither[TypeLeft, TypeResult]:
            """Maps the applicative's function, curried, over this AsyncEither's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: AsyncEither[TypeLeft, Callable[..., TypeResult]],
        applicative_value: AsyncEither[TypeLeft, Any],
    ) -> AsyncEither[TypeLeft, TypeResult]:
        """AsyncEither applicative interface for AsyncEithers containing a function (<*>).

        Parameters
        ----------
        applicative_value: AsyncEither[TypeLeft, TypePure]
            AsyncEither value which will be applied to the AsyncEither containing a function.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeResult]
            Applies an AsyncEither containing a function to an AsyncEither of type TypePure
            (value or function).
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncEither[TypeLeft, TypeResult]:
            """Maps the curried applicative function, held by this AsyncEither, over `applicative_value`."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def and_(self, other: AsyncEither[TypeLeft, TypeResult]) -> AsyncEither[TypeLeft, TypeResult]:
        """Returns `other` if this AsyncEither resolves to Right, otherwise the original Left.

        Parameters
        ----------
        other: AsyncEither[TypeLeft, TypeResult]
            AsyncEither to be returned if this AsyncEither resolves to Right.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeResult]
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncEither[Any, TypeRight]) -> AsyncEither[Any, TypeRight]:
        """Returns this AsyncEither if it resolves to Right, otherwise `other`.

        Parameters
        ----------
        other: AsyncEither[Any, TypeRight]
            AsyncEither to be returned if this AsyncEither resolves to Left.

        Returns
        -------
        async_either: AsyncEither[Any, TypeRight]
        """

        async def run() -> Either[Any, TypeRight]:
            """Awaits self, falling back to `other` if this AsyncEither resolves to Left."""
            either = await self
            if isinstance(either, Right):
                return either
            return await other

        return AsyncEither(run)

    def zip(
        self, other: AsyncEither[TypeLeft, TypePure]
    ) -> AsyncEither[TypeLeft, Tuple[TypeRight, TypePure]]:
        """Combines this AsyncEither with another into an AsyncEither of a tuple, or Left if either is Left.

        Parameters
        ----------
        other: AsyncEither[TypeLeft, TypePure]
            AsyncEither to be zipped with this AsyncEither.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, Tuple[TypeRight, TypePure]]
        """

        async def run() -> Either[TypeLeft, Tuple[TypeRight, TypePure]]:
            """Awaits both self and `other`, combining their values if both are Right."""
            either = await self
            if isinstance(either, Left):
                return either
            other_either = await other
            return other_either.map(lambda o: (either.unwrap(), o))

        return AsyncEither(run)

    def flatten(
        self: AsyncEither[TypeLeft, AsyncEither[TypeLeft, TypeResult]],
    ) -> AsyncEither[TypeLeft, TypeResult]:
        """Flattens a nested AsyncEither by one level.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeResult]
            Returns the nested AsyncEither's eventual result, or the original Left without
            awaiting it if this AsyncEither resolves to Left.
        """

        async def run() -> Either[TypeLeft, TypeResult]:
            """Awaits self, then awaits the nested AsyncEither if Right."""
            either = await self
            if isinstance(either, Left):
                return either
            return await either.unwrap()

        return AsyncEither(run)

    def inspect(
        self, function: Callable[[TypeRight], Union[None, Awaitable[None]]]
    ) -> AsyncEither[TypeLeft, TypeRight]:
        """Inspect the AsyncEither's resolved value of TypeRight.

        Parameters
        ----------
        function: Callable[[TypeRight], Union[None, Awaitable[None]]]
            Sync or async inspection function called with the resolved value if Right.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeRight]
        """

        async def run() -> Either[TypeLeft, TypeRight]:
            """Awaits self, calling `function` for its side effect only if Right."""
            either = await self
            if isinstance(either, Right):
                await _resolve(function(either.unwrap()))
            return either

        return AsyncEither(run)

    def inspect_left(
        self, function: Callable[[TypeLeft], Union[None, Awaitable[None]]]
    ) -> AsyncEither[TypeLeft, TypeRight]:
        """Inspect the AsyncEither's resolved value of TypeLeft.

        Parameters
        ----------
        function: Callable[[TypeLeft], Union[None, Awaitable[None]]]
            Sync or async inspection function called with the resolved value if Left.

        Returns
        -------
        async_either: AsyncEither[TypeLeft, TypeRight]
        """

        async def run() -> Either[TypeLeft, TypeRight]:
            """Awaits self, calling `function` for its side effect only if Left."""
            either = await self
            match either:
                case Left(value):
                    await _resolve(function(value))
                case _:
                    pass
            return either

        return AsyncEither(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncEither."""
        return f"AsyncEither({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncEither (same as __str__)."""
        return str(self)
