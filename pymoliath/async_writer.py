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
from pymoliath.util import curry
from pymoliath.writer import TypeMonoid, Writer

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypeInner = TypeVar("TypeInner")


class AsyncWriter(Generic[TypeSource, TypeMonoid]):
    """Async Writer Monad: a deferred computation which resolves to a Tuple[TypeSource, TypeMonoid]
    once awaited.

    The AsyncWriter[TypeSource, TypeMonoid] class represents a deferred computation that produces
    a tuple containing a value of type TypeSource and one of type TypeMonoid. The value TypeMonoid
    represents a data type which must follow the monoid laws. A typical example of an TypeMonoid
    value could be a logging String.

    TypeSource: any type which will be used for monad computation (bind, map, apply)
    TypeMonoid: any type that behaves as a monoid which can be added together.

    The class of monoids TypeMonoid (types with an associative binary operation (e.g. + ) that has
    an identity).

    Monoid instances should satisfy the following laws:

    1. Closure: If 'a' and 'b' are in TypeMonoid, then 'a + b' is also in TypeMonoid.
    2. Identity: There exists an element in TypeMonoid (denoted 0) such that: a + 0 = a = 0 + a
    3. Associativity: (a + b) + c = a + (b + c)

    Directly awaitable - nothing in a chain of map/bind/tell/... runs until the AsyncWriter itself
    is awaited (`await an_async_writer`), mirroring the deferred-pipeline design used by AsyncMaybe
    (pymoliath/async_maybe.py). Unlike AsyncMaybe there is no failure state to preserve, so awaiting
    resolves directly to the raw Tuple[TypeSource, TypeMonoid] - exactly what sync Writer.run()
    already returns - rather than to a wrapper object.

    Callbacks passed to map/bind may be plain sync functions or `async def` functions - whichever
    is returned is auto-detected at the point it's called, so real async I/O can be mixed freely
    with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[], Awaitable[Tuple[TypeSource, TypeMonoid]]]) -> None:
        """AsyncWriter constructor which takes a zero-argument async callable resolving to a
        Tuple[TypeSource, TypeMonoid].

        Parameters
        ----------
        run: Callable[[], Awaitable[Tuple[TypeSource, TypeMonoid]]]
            Zero-argument callable returning a fresh awaitable each call, resolving to the
            (value, monoid) pair. To stay re-awaitable, `run` must produce a *new* awaitable every
            call rather than handing back an already-created (and possibly already-consumed)
            coroutine object.
        """
        self._run = run

    def __await__(self) -> Generator[Any, None, Tuple[TypeSource, TypeMonoid]]:
        """Runs the pipeline and resolves to the final Tuple[TypeSource, TypeMonoid]."""
        return self._run().__await__()

    @staticmethod
    def from_value(value: TypeSource, monoid: TypeMonoid) -> AsyncWriter[TypeSource, TypeMonoid]:
        """Lifts an already-resolved value and monoid into an AsyncWriter.

        Parameters
        ----------
        value: TypeSource
            Generic writer monad value.
        monoid: TypeMonoid
            Generic writer monad monoid (see class docstring).

        Returns
        -------
        async_writer: AsyncWriter[TypeSource, TypeMonoid]
        """

        async def run() -> Tuple[TypeSource, TypeMonoid]:
            """Resolves immediately to (value, monoid)."""
            return value, monoid

        return AsyncWriter(run)

    @staticmethod
    def from_writer(writer: Writer[TypeSource, TypeMonoid]) -> AsyncWriter[TypeSource, TypeMonoid]:
        """Lifts an existing sync Writer into an AsyncWriter.

        Parameters
        ----------
        writer: Writer[TypeSource, TypeMonoid]
            Writer to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_writer: AsyncWriter[TypeSource, TypeMonoid]
        """

        async def run() -> Tuple[TypeSource, TypeMonoid]:
            """Resolves immediately to `writer`'s (value, monoid) pair."""
            return writer.run()

        return AsyncWriter(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[Tuple[TypeSource, TypeMonoid]]],
    ) -> AsyncWriter[TypeSource, TypeMonoid]:
        """Wraps a zero-argument async callable producing a (value, monoid) pair.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[Tuple[TypeSource, TypeMonoid]]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncWriter stays
            re-awaitable).

        Returns
        -------
        async_writer: AsyncWriter[TypeSource, TypeMonoid]
        """

        async def run() -> Tuple[TypeSource, TypeMonoid]:
            """Awaits `coroutine_function` and passes its result through unchanged."""
            return await coroutine_function()

        return AsyncWriter(run)

    def map(
        self, function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncWriter[TypeResult, TypeMonoid]:
        """AsyncWriter functor interface (>=, map).

        Definition: M(a, m) >= f: a -> b => M(b, m)

        Parameters
        ----------
        function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function applied to the resolved value.

        Returns
        -------
        async_writer: AsyncWriter[TypeResult, TypeMonoid]
            Returns a new AsyncWriter with the function result as value and the monoid unchanged.
        """

        async def run() -> Tuple[TypeResult, TypeMonoid]:
            """Awaits self, then applies `function` to the resolved value."""
            value, monoid = await self
            return await _resolve(function(value)), monoid

        return AsyncWriter(run)

    def bind(
        self,
        function: Callable[
            [TypeSource],
            Union[
                AsyncWriter[TypeResult, TypeMonoid],
                Writer[TypeResult, TypeMonoid],
                Awaitable[Tuple[TypeResult, TypeMonoid]],
            ],
        ],
    ) -> AsyncWriter[TypeResult, TypeMonoid]:
        """AsyncWriter bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], AsyncWriter[TypeResult, TypeMonoid] | Writer[TypeResult, TypeMonoid] | Awaitable[Tuple[TypeResult, TypeMonoid]]]
            Function applied to the resolved value, returning another AsyncWriter, a plain Writer,
            or an awaitable resolving to a (value, monoid) pair - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_writer: AsyncWriter[TypeResult, TypeMonoid]
            Returns a new AsyncWriter with the function result and the closure of the monoids (+).
        """

        async def run() -> Tuple[TypeResult, TypeMonoid]:
            """Awaits self, then chains into `function`'s result, combining the monoids."""
            value, monoid = await self
            result = function(value)
            if isinstance(result, AsyncWriter):
                other_value, other_monoid = await result
            elif isinstance(result, Writer):
                other_value, other_monoid = result.run()
            else:
                other_value, other_monoid = await _resolve(result)
            return other_value, monoid + other_monoid

        return AsyncWriter(run)

    def apply(
        self, applicative: AsyncWriter[Callable[..., TypeResult], TypeMonoid]
    ) -> AsyncWriter[TypeResult, TypeMonoid]:
        """AsyncWriter applicative interface for AsyncWriters containing a value.

        Parameters
        ----------
        applicative: AsyncWriter[Callable[[TypeSource], TypeResult], TypeMonoid]
            AsyncWriter applicative containing a function as value.

        Returns
        -------
        async_writer: AsyncWriter[TypeResult, TypeMonoid]
            Applies an AsyncWriter containing a value of type TypeSource to an AsyncWriter
            containing a function of type Callable[[TypeSource], TypeResult].
        """

        async def run() -> Tuple[TypeResult, TypeMonoid]:
            """Awaits both self and `applicative`, applying the function to the value."""
            value, monoid = await self
            function, other_monoid = await applicative
            # The dynamic partial-application fallback can't be typed statically:
            # partial[TypeResult] isn't TypeResult, but it's a valid TypeResult once
            # fully applied by a later apply/apply2 call.
            return curry(function)(value), monoid + other_monoid

        return AsyncWriter(run)

    def apply2(
        self: AsyncWriter[Callable[..., TypeResult], TypeMonoid],
        monad_value: AsyncWriter[Any, TypeMonoid],
    ) -> AsyncWriter[TypeResult, TypeMonoid]:
        """AsyncWriter applicative interface for AsyncWriters containing a function.

        Parameters
        ----------
        monad_value: AsyncWriter[TypePure, TypeMonoid]
            AsyncWriter value which will be applied to the AsyncWriter containing a function.

        Returns
        -------
        async_writer: AsyncWriter[TypeResult, TypeMonoid]
            Applies an AsyncWriter containing a function of type Callable[[TypeSource], TypeResult]
            to an AsyncWriter of type TypeSource (value or function).
        """

        async def run() -> Tuple[TypeResult, TypeMonoid]:
            """Awaits both self and `monad_value`, applying the function to the value."""
            value_function, monoid = await self
            value, other_monoid = await monad_value
            return curry(value_function)(value), monoid + other_monoid

        return AsyncWriter(run)

    def tell(self, monoid_value: TypeMonoid) -> AsyncWriter[TypeSource, TypeMonoid]:
        """AsyncWriter specific function to add a monoid value to the log.

        Definition: Monad(a, m) :: tell(n) -> Monad(a, m + n) where m must be a monoid which can be
        empty.

        Parameters
        ----------
        monoid_value: TypeMonoid
            Monoid value of type TypeMonoid.

        Returns
        -------
        async_writer: AsyncWriter[TypeSource, TypeMonoid]
           Returns an AsyncWriter containing the closure (+) of the passed monoid value.
        """

        async def run() -> Tuple[TypeSource, TypeMonoid]:
            """Awaits self, then appends `monoid_value` to the resolved monoid."""
            value, monoid = await self
            return value, monoid + monoid_value

        return AsyncWriter(run)

    def listen(
        self,
    ) -> AsyncWriter[Tuple[TypeSource, TypeMonoid], TypeMonoid]:
        """AsyncWriter specific function listen.

        Definition: listen :: Monad(a, m) -> Monad(a, (a, m))

        Listen is an action that executes the action in the monad and adds its output to the value
        of the computation.

        Returns
        -------
        async_writer: AsyncWriter[Tuple[TypeSource, TypeMonoid], TypeMonoid]
        """

        async def run() -> Tuple[Tuple[TypeSource, TypeMonoid], TypeMonoid]:
            """Awaits self, pairing the resolved value with its own (value, monoid) pair."""
            resolved = await self
            _, monoid = resolved
            return resolved, monoid

        return AsyncWriter(run)

    def pass_(
        self: AsyncWriter[
            Tuple[TypeInner, Callable[[TypeMonoid], TypeMonoid]], TypeMonoid
        ],
    ) -> AsyncWriter[TypeInner, TypeMonoid]:
        """AsyncWriter specific function pass_ (actually pass).

        Definition: pass :: Monad((a, f: m -> m), m) -> Monad(a, m)

        The pass function will execute the function which is contained in the tuple value of the
        AsyncWriter and applies it to the monoid value. Returns an AsyncWriter containing the value
        and the resulting monoid value.

        Returns
        -------
        async_writer: AsyncWriter[TypeInner, TypeMonoid]
        """

        async def run() -> Tuple[TypeInner, TypeMonoid]:
            """Awaits self, then applies the carried monoid function to the resolved monoid."""
            pass_tuple, monoid = await self
            value, monoid_function = pass_tuple
            return value, monoid_function(monoid)

        return AsyncWriter(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncWriter."""
        return f"AsyncWriter({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncWriter (same as __str__)."""
        return str(self)
