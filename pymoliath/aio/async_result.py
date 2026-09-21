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

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.result import Err, Ok, Result
from pymoliath.util import curry

TypeOk = TypeVar("TypeOk")
TypeErr = TypeVar("TypeErr")
TypeReturn = TypeVar("TypeReturn")
TypePure = TypeVar("TypePure")


class AsyncResult(Generic[TypeOk, TypeErr]):
    """Async Result Monad: a deferred computation which resolves to a Result[TypeOk, TypeErr] once awaited.

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

    def __init__(self, run: Callable[[], Awaitable[Result[TypeOk, TypeErr]]]) -> None:
        """AsyncResult constructor which takes a zero-argument async callable resolving to a Result.

        Parameters
        ----------
        run: Callable[[], Awaitable[Result[TypeOk, TypeErr]]]
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

    def __await__(self) -> Generator[Any, None, Result[TypeOk, TypeErr]]:
        """Runs the pipeline and resolves to the final Result[TypeOk, TypeErr].

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(10))
        Ok(10)
        """
        return self._run().__await__()

    @staticmethod
    def from_ok(value: TypeOk) -> AsyncResult[TypeOk, Any]:
        """Lifts a plain value into an already-Ok AsyncResult.

        Parameters
        ----------
        value: TypeOk
            Value to be wrapped as Ok once awaited.

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(10))
        Ok(10)
        """

        async def run() -> Result[TypeOk, Any]:
            """Resolves immediately to Ok(value)."""
            return Ok(value)

        return AsyncResult(run)

    @staticmethod
    def from_err(value: TypeErr) -> AsyncResult[Any, TypeErr]:
        """Lifts a plain value into an already-Err AsyncResult.

        Parameters
        ----------
        value: TypeErr
            Value to be wrapped as Err once awaited.

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_err("error"))
        Err(error)
        """

        async def run() -> Result[Any, TypeErr]:
            """Resolves immediately to Err(value)."""
            return Err(value)

        return AsyncResult(run)

    @staticmethod
    def from_result(result: Result[TypeOk, TypeErr]) -> AsyncResult[TypeOk, TypeErr]:
        """Lifts an existing sync Result (Ok or Err) into an AsyncResult.

        Parameters
        ----------
        result: Result[TypeOk, TypeErr]
            Result to be wrapped, resolved unchanged once awaited.

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_result(Ok(10)))
        Ok(10)
        >>> asyncio.run(AsyncResult.from_result(Err("error")))
        Err(error)
        """

        async def run() -> Result[TypeOk, TypeErr]:
            """Resolves immediately to `result`."""
            return result

        return AsyncResult(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[], Awaitable[TypeOk]],
    ) -> AsyncResult[TypeOk, Any]:
        """Wraps a zero-argument async callable producing a raw value as an Ok once awaited.

        Parameters
        ----------
        coroutine_function: Callable[[], Awaitable[TypeOk]]
            Zero-argument callable returning a fresh awaitable each call (e.g. an `async def`
            function, not an already-created coroutine object, so the AsyncResult stays
            re-awaitable).

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> async def fetch_ten() -> int: return 10
        >>> asyncio.run(AsyncResult.from_coroutine(fetch_ten))
        Ok(10)
        """

        async def run() -> Result[TypeOk, Any]:
            """Awaits `coroutine_function` and wraps its result as Ok."""
            return Ok(await coroutine_function())

        return AsyncResult(run)

    def map(
        self, function: Callable[[TypeOk], Union[TypeReturn, Awaitable[TypeReturn]]]
    ) -> AsyncResult[TypeReturn, TypeErr]:
        """AsyncResult functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeOk], Union[TypeReturn, Awaitable[TypeReturn]]]
            Sync or async function applied to the resolved value if Ok.

        Returns
        -------
        async_result: AsyncResult[TypeReturn, TypeErr]
            Returns a new AsyncResult which resolves to Ok with the function result, or Err
            without calling `function`, if this AsyncResult resolves to Err.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(5).map(lambda x: x + 1))
        Ok(6)
        >>> asyncio.run(AsyncResult.from_err("error").map(lambda x: x + 1))
        Err(error)
        """

        async def run() -> Result[TypeReturn, TypeErr]:
            """Awaits self, then applies `function`, short-circuiting on Err."""
            outcome = await self
            if isinstance(outcome, Err):
                return outcome
            return Ok(await _resolve(function(outcome.unwrap())))

        return AsyncResult(run)

    def map_err(
        self, function: Callable[[TypeErr], Union[TypeReturn, Awaitable[TypeReturn]]]
    ) -> AsyncResult[TypeOk, TypeReturn]:
        """AsyncResult functor interface for the Err channel.

        Parameters
        ----------
        function: Callable[[TypeErr], Union[TypeReturn, Awaitable[TypeReturn]]]
            Sync or async function applied to the resolved value if Err.

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeReturn]
            Returns a new AsyncResult which resolves to Err with the function result, or Ok
            without calling `function`, if this AsyncResult resolves to Ok.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_err("error").map_err(str.upper))
        Err(ERROR)
        >>> asyncio.run(AsyncResult.from_ok(10).map_err(str.upper))
        Ok(10)
        """

        async def run() -> Result[TypeOk, TypeReturn]:
            """Awaits self, then applies `function` to the Err value, short-circuiting on Ok."""
            outcome = await self
            match outcome:
                case Ok():
                    return outcome
                case Err(err_value):
                    return Err(await _resolve(function(err_value)))

        return AsyncResult(run)

    def bind(
        self,
        function: Callable[
            [TypeOk],
            Union[
                AsyncResult[TypeReturn, TypeErr],
                Result[TypeReturn, TypeErr],
                Awaitable[Result[TypeReturn, TypeErr]],
            ],
        ],
    ) -> AsyncResult[TypeReturn, TypeErr]:
        """AsyncResult bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeOk], AsyncResult[TypeReturn, TypeErr] | Result[TypeReturn, TypeErr] | Awaitable[Result[TypeReturn, TypeErr]]]
            Function applied to the resolved value if Ok, returning another AsyncResult, a plain
            Result, or an awaitable resolving to a Result - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_result: AsyncResult[TypeReturn, TypeErr]
            Returns a new AsyncResult with the function result if Ok, otherwise Err without
            calling `function`.

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(5).bind(lambda x: AsyncResult.from_ok(x + 1)))
        Ok(6)
        >>> asyncio.run(AsyncResult.from_err("error").bind(lambda x: AsyncResult.from_ok(x + 1)))
        Err(error)
        """

        async def run() -> Result[TypeReturn, TypeErr]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Err."""
            outcome = await self
            if isinstance(outcome, Err):
                return outcome
            next_result = function(outcome.unwrap())
            if isinstance(next_result, AsyncResult):
                return await next_result
            return await _resolve(next_result)

        return AsyncResult(run)

    def bind_err(
        self,
        function: Callable[
            [TypeErr],
            Union[
                AsyncResult[TypeOk, TypeReturn],
                Result[TypeOk, TypeReturn],
                Awaitable[Result[TypeOk, TypeReturn]],
            ],
        ],
    ) -> AsyncResult[TypeOk, TypeReturn]:
        """AsyncResult bind interface for the Err channel.

        Parameters
        ----------
        function: Callable[[TypeErr], AsyncResult[TypeOk, TypeReturn] | Result[TypeOk, TypeReturn] | Awaitable[Result[TypeOk, TypeReturn]]]
            Function applied to the resolved value if Err, returning another AsyncResult, a plain
            Result, or an awaitable resolving to a Result - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeReturn]
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

        async def run() -> Result[TypeOk, TypeReturn]:
            """Awaits self, then chains into `function`'s result, short-circuiting on Ok."""
            outcome = await self
            match outcome:
                case Ok():
                    return outcome
                case Err(err_value):
                    next_result = function(err_value)
                    if isinstance(next_result, AsyncResult):
                        return await next_result
                    return await _resolve(next_result)

        return AsyncResult(run)

    def apply(
        self, applicative: AsyncResult[Callable[..., TypeReturn], TypeErr]
    ) -> AsyncResult[TypeReturn, TypeErr]:
        """AsyncResult applicative interface for AsyncResults containing a value (<*>).

        Parameters
        ----------
        applicative: AsyncResult[TypeApplicative, TypeErr] (TypeApplicative: any callable type)
            Applicative AsyncResult which contains a function and will be applied to the
            AsyncResult containing a value.

        Returns
        -------
        async_result: AsyncResult[TypeReturn, TypeErr]
            Applies an AsyncResult containing a value of type TypeOk to an AsyncResult containing
            a function.

        Examples
        --------
        >>> import asyncio
        >>> val = AsyncResult.from_ok(10)
        >>> func = AsyncResult.from_ok(lambda x: x * 2)
        >>> asyncio.run(val.apply(func))
        Ok(20)
        """

        def binder(
            applicative_function: Callable[..., TypeReturn],
        ) -> AsyncResult[TypeReturn, TypeErr]:
            """Maps the applicative's function, curried, over this AsyncResult's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: AsyncResult[Callable[..., TypeReturn], TypeErr],
        applicative_value: AsyncResult[Any, TypeErr],
    ) -> AsyncResult[TypeReturn, TypeErr]:
        """AsyncResult applicative interface for AsyncResults containing a function (<*>).

        Parameters
        ----------
        applicative_value: AsyncResult[TypePure, TypeErr]
            AsyncResult value which will be applied to the AsyncResult containing a function.

        Returns
        -------
        async_result: AsyncResult[TypeReturn, TypeErr]
            Applies an AsyncResult containing a function to an AsyncResult of type TypePure (value
            or function).

        Examples
        --------
        >>> import asyncio
        >>> func = AsyncResult.from_ok(lambda x: x * 2)
        >>> val = AsyncResult.from_ok(10)
        >>> asyncio.run(func.apply2(val))
        Ok(20)
        """

        def binder(
            applicative_function: Callable[..., TypeReturn],
        ) -> AsyncResult[TypeReturn, TypeErr]:
            """Maps the curried applicative function, held by this AsyncResult, over `applicative_value`."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def and_(
        self, other: AsyncResult[TypeReturn, TypeErr]
    ) -> AsyncResult[TypeReturn, TypeErr]:
        """Returns `other` if this AsyncResult resolves to Ok, otherwise Err.

        Parameters
        ----------
        other: AsyncResult[TypeReturn, TypeErr]
            AsyncResult to be returned if this AsyncResult resolves to Ok.

        Returns
        -------
        async_result: AsyncResult[TypeReturn, TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(1).and_(AsyncResult.from_ok(2)))
        Ok(2)
        >>> asyncio.run(AsyncResult.from_err("error").and_(AsyncResult.from_ok(2)))
        Err(error)
        """
        return self.bind(lambda _: other)

    def or_(self, other: AsyncResult[TypeOk, TypeErr]) -> AsyncResult[TypeOk, TypeErr]:
        """Returns this AsyncResult if it resolves to Ok, otherwise `other`.

        Parameters
        ----------
        other: AsyncResult[TypeOk, TypeErr]
            AsyncResult to be returned if this AsyncResult resolves to Err.

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(1).or_(AsyncResult.from_ok(2)))
        Ok(1)
        >>> asyncio.run(AsyncResult.from_err("error").or_(AsyncResult.from_ok(2)))
        Ok(2)
        """

        async def run() -> Result[TypeOk, TypeErr]:
            """Awaits self, falling back to `other` if this AsyncResult resolves to Err."""
            outcome = await self
            if isinstance(outcome, Ok):
                return outcome
            return await other

        return AsyncResult(run)

    def zip(
        self, other: AsyncResult[TypePure, TypeErr]
    ) -> AsyncResult[Tuple[TypeOk, TypePure], TypeErr]:
        """Combines this AsyncResult with another into an AsyncResult of a tuple, or Err if either is Err.

        Parameters
        ----------
        other: AsyncResult[TypePure, TypeErr]
            AsyncResult to be zipped with this AsyncResult.

        Returns
        -------
        async_result: AsyncResult[Tuple[TypeOk, TypePure], TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(1).zip(AsyncResult.from_ok(2)))
        Ok((1, 2))
        >>> asyncio.run(AsyncResult.from_err("error").zip(AsyncResult.from_ok(2)))
        Err(error)
        """

        async def run() -> Result[Tuple[TypeOk, TypePure], TypeErr]:
            """Awaits both self and `other`, combining their values if both are Ok."""
            outcome = await self
            if isinstance(outcome, Err):
                return outcome
            other_outcome = await other
            return other_outcome.map(lambda o: (outcome.unwrap(), o))

        return AsyncResult(run)

    def flatten(
        self: AsyncResult[AsyncResult[TypeReturn, TypeErr], TypeErr],
    ) -> AsyncResult[TypeReturn, TypeErr]:
        """Flattens a nested AsyncResult by one level.

        Returns
        -------
        async_result: AsyncResult[TypeReturn, TypeErr]
            Returns the nested AsyncResult's eventual result, or Err without awaiting it if this
            AsyncResult resolves to Err.

        Examples
        --------
        >>> import asyncio
        >>> nested = AsyncResult.from_ok(AsyncResult.from_ok(1))
        >>> asyncio.run(nested.flatten())
        Ok(1)
        """

        async def run() -> Result[TypeReturn, TypeErr]:
            """Awaits self, then awaits the nested AsyncResult if Ok."""
            outcome = await self
            if isinstance(outcome, Err):
                return outcome
            return await outcome.unwrap()

        return AsyncResult(run)

    def inspect(
        self, function: Callable[[TypeOk], Union[None, Awaitable[None]]]
    ) -> AsyncResult[TypeOk, TypeErr]:
        """Inspect the AsyncResult's resolved value of TypeOk.

        Parameters
        ----------
        function: Callable[[TypeOk], Union[None, Awaitable[None]]]
            Sync or async inspection function called with the resolved value if Ok.

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_ok(10).inspect(lambda x: print(f"Value: {x}")))
        Value: 10
        Ok(10)
        """

        async def run() -> Result[TypeOk, TypeErr]:
            """Awaits self, calling `function` for its side effect only if Ok."""
            outcome = await self
            if isinstance(outcome, Ok):
                await _resolve(function(outcome.unwrap()))
            return outcome

        return AsyncResult(run)

    def inspect_err(
        self, function: Callable[[TypeErr], Union[None, Awaitable[None]]]
    ) -> AsyncResult[TypeOk, TypeErr]:
        """Inspect the AsyncResult's resolved value of TypeErr.

        Parameters
        ----------
        function: Callable[[TypeErr], Union[None, Awaitable[None]]]
            Sync or async inspection function called with the resolved value if Err.

        Returns
        -------
        async_result: AsyncResult[TypeOk, TypeErr]

        Examples
        --------
        >>> import asyncio
        >>> asyncio.run(AsyncResult.from_err("error").inspect_err(lambda e: print(f"Error: {e}")))
        Error: error
        Err(error)
        """

        async def run() -> Result[TypeOk, TypeErr]:
            """Awaits self, calling `function` for its side effect only if Err."""
            outcome = await self
            match outcome:
                case Err(err_value):
                    await _resolve(function(err_value))
                case Ok():
                    pass
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
