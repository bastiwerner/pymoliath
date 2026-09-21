"""
.. include:: ../docs/reader/README.md
   :start-after: ## AsyncReader
"""

from __future__ import annotations

from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Type,
    TypeVar,
    Union,
)

from pymoliath.aio.utils import resolve as _resolve
from pymoliath.reader import Reader
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeEnv = TypeVar("TypeEnv")
TypeResult = TypeVar("TypeResult")


class AsyncReader(Generic[TypeEnv, TypeSource]):
    """Async Reader Monad: a deferred, environment-parameterized computation.

    Unlike AsyncMaybe (pymoliath/async_maybe.py), AsyncReader is NOT bare-`await`-able: like its
    sync counterpart Reader (pymoliath/reader.py), it needs an environment value to run - Python's
    `__await__` protocol takes no arguments, so there's no way to `await an_async_reader` without
    supplying one. Instead, call `await an_async_reader.run(env)`, mirroring sync `Reader.run(env)`
    exactly but returning an awaitable instead of running immediately. Nothing in a chain of
    map/bind/apply/local runs until `.run(env)` is awaited.

    Callbacks passed to map/bind may be plain sync functions or `async def` functions - whichever
    is returned is auto-detected at the point it's called, so real async I/O can be mixed freely
    with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(self, run: Callable[[TypeEnv], Awaitable[TypeSource]]) -> None:
        """AsyncReader constructor which takes an env-accepting async callable.

        Parameters
        ----------
        run: Callable[[TypeEnv], Awaitable[TypeSource]]
            Callable which, given an environment, returns a fresh awaitable each call, resolving
            to the reader's value. To stay re-runnable, `run` must produce a *new* awaitable every
            call rather than handing back an already-created (and possibly already-consumed)
            coroutine object.
        """
        self._run = run

    async def run(self, env: TypeEnv) -> TypeSource:
        """Runs the pipeline with `env` and resolves to the final TypeSource value.

        Definition: AsyncReader(f: env -> a).run(env) => a

        Parameters
        ----------
        env: TypeEnv
            Environment value to run this AsyncReader with.

        Returns
        -------
        result: TypeSource
        """
        return await self._run(env)

    @staticmethod
    def from_value(value: TypeSource) -> AsyncReader[Any, TypeSource]:
        """Lifts a plain value into an AsyncReader which ignores the environment.

        Parameters
        ----------
        value: TypeSource
            Value to be resolved to regardless of the environment passed to `run`.

        Returns
        -------
        async_reader: AsyncReader[Any, TypeSource]
        """

        async def run(env: Any) -> TypeSource:
            """Resolves immediately to `value`, ignoring `env`."""
            return value

        return AsyncReader(run)

    @staticmethod
    def from_reader(
        reader: Reader[TypeEnv, TypeSource],
    ) -> AsyncReader[TypeEnv, TypeSource]:
        """Lifts an existing sync Reader into an AsyncReader.

        Parameters
        ----------
        reader: Reader[TypeEnv, TypeSource]
            Reader to be wrapped, run synchronously (unchanged) once `.run(env)` is awaited.

        Returns
        -------
        async_reader: AsyncReader[TypeEnv, TypeSource]
        """

        async def run(env: TypeEnv) -> TypeSource:
            """Resolves immediately to `reader.run(env)`."""
            return reader.run(env)

        return AsyncReader(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[TypeEnv], Awaitable[TypeSource]],
    ) -> AsyncReader[TypeEnv, TypeSource]:
        """Wraps an env-accepting async callable as an AsyncReader.

        Parameters
        ----------
        coroutine_function: Callable[[TypeEnv], Awaitable[TypeSource]]
            Callable which, given an environment, returns a fresh awaitable each call (e.g. an
            `async def` function, not an already-created coroutine object, so the AsyncReader
            stays re-runnable).

        Returns
        -------
        async_reader: AsyncReader[TypeEnv, TypeSource]
        """
        return AsyncReader(coroutine_function)

    def map(
        self, function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncReader[TypeEnv, TypeResult]:
        """AsyncReader functor interface (>=, map).

        Definition: Reader(f: e -> a) >= f: a -> b => Reader(f: e -> b)

        Parameters
        ----------
        function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function applied to the resolved value.

        Returns
        -------
        async_reader: AsyncReader[TypeEnv, TypeResult]
            Returns a new AsyncReader with the function result as value.
        """

        async def run(env: TypeEnv) -> TypeResult:
            """Runs self with `env`, then applies `function` to the result."""
            return await _resolve(function(await self.run(env)))

        return AsyncReader(run)

    def bind(
        self,
        function: Callable[
            [TypeSource],
            Union[
                AsyncReader[TypeEnv, TypeResult],
                Reader[TypeEnv, TypeResult],
                Awaitable[TypeResult],
            ],
        ],
    ) -> AsyncReader[TypeEnv, TypeResult]:
        """AsyncReader bind interface (>>=, bind, flatMap).

        Definition: Reader(f: e -> a) >>= f: a -> Reader(f: e -> b) => Reader(f: e -> b)

        Parameters
        ----------
        function: Callable[[TypeSource], AsyncReader[TypeEnv, TypeResult] | Reader[TypeEnv, TypeResult] | Awaitable[TypeResult]]
            Function applied to the resolved value, returning another AsyncReader, a plain
            Reader, or an awaitable resolving to a value - whichever shape is returned is
            auto-detected.

        Returns
        -------
        async_reader: AsyncReader[TypeEnv, TypeResult]
            Returns a new AsyncReader with the function result.
        """

        async def run(env: TypeEnv) -> TypeResult:
            """Runs self with `env`, then chains into `function`'s result, run with the same `env`."""
            result = function(await self.run(env))
            if isinstance(result, AsyncReader):
                return await result.run(env)
            if isinstance(result, Reader):
                return result.run(env)
            return await _resolve(result)

        return AsyncReader(run)

    def apply(
        self, applicative: AsyncReader[TypeEnv, Callable[..., TypeResult]]
    ) -> AsyncReader[TypeEnv, TypeResult]:
        """AsyncReader applicative interface for AsyncReaders containing a value (<*>).

        Definition: Reader(f: e -> a) <*> Reader(f: e -> f: a -> b) => Reader(f: e -> b)

        Parameters
        ----------
        applicative: AsyncReader[TypeEnv, Callable[[TypeSource], TypeResult]]
            Applicative AsyncReader which contains a function and will be applied to the
            AsyncReader containing a value.

        Returns
        -------
        async_reader: AsyncReader[TypeEnv, TypeResult]
            Applies an AsyncReader containing a value to an AsyncReader containing a function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncReader[TypeEnv, TypeResult]:
            """Maps the applicative's function, curried, over this AsyncReader's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: AsyncReader[TypeEnv, Callable[..., TypeResult]],
        applicative_value: AsyncReader[TypeEnv, Any],
    ) -> AsyncReader[TypeEnv, TypeResult]:
        """AsyncReader applicative interface for AsyncReaders containing a function (<*>).

        Definition: Reader(f: e -> f: a -> b) <*> Reader(f: e -> a) => Reader(f: e -> b)

        Parameters
        ----------
        applicative_value: AsyncReader[TypeEnv, TypeSource]
            AsyncReader value which will be applied to the AsyncReader containing a function.

        Returns
        -------
        async_reader: AsyncReader[TypeEnv, TypeResult]
            Applies an AsyncReader containing a function to an AsyncReader with a value or
            function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncReader[TypeEnv, TypeResult]:
            """Maps the applicative value's function, curried, over this AsyncReader's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    @classmethod
    def ask(cls: Type[AsyncReader[TypeEnv, TypeEnv]]) -> AsyncReader[TypeEnv, TypeEnv]:
        """AsyncReader special function ask to return an AsyncReader which resolves to the environment.

        Definition: AsyncReader.ask() => Reader(f: env -> env)

        Returns
        -------
        async_reader: AsyncReader[TypeEnv, TypeEnv]
        """

        async def run(env: TypeEnv) -> TypeEnv:
            """Resolves immediately to `env`, unchanged."""
            return env

        return cls(run)

    def local(
        self, function: Callable[[TypeEnv], TypeEnv]
    ) -> AsyncReader[TypeEnv, TypeSource]:
        """AsyncReader specific function local. Modifies the environment before it reaches this reader.

        Parameters
        ----------
        function: Callable[[TypeEnv], TypeEnv]
            The function to modify the environment.

        Returns
        -------
        async_reader: AsyncReader[TypeEnv, TypeSource]
            Returns a new AsyncReader which runs this one with the modified environment.
        """

        async def run(env: TypeEnv) -> TypeSource:
            """Runs self with `function(env)` instead of `env`."""
            return await self.run(function(env))

        return AsyncReader(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncReader."""
        return f"AsyncReader({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncReader (same as __str__)."""
        return str(self)
