from __future__ import annotations

from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pymoliath._async import resolve as _resolve
from pymoliath.state import State
from pymoliath.util import curry

TypeState = TypeVar("TypeState")
TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")


class AsyncState(Generic[TypeState, TypeSource]):
    """Async State Monad: a deferred, state-parameterized computation.

    Unlike AsyncMaybe (pymoliath/async_maybe.py), AsyncState is NOT bare-`await`-able: like its
    sync counterpart State (pymoliath/state.py), it needs an initial state value to run - Python's
    `__await__` protocol takes no arguments, so there's no way to `await an_async_state` without
    supplying one. Instead, call `await an_async_state.run(state)`, mirroring sync
    `State.run(state)` exactly but returning an awaitable instead of running immediately. Nothing
    in a chain of map/bind/apply runs until `.run(state)` is awaited.

    Callbacks passed to map/bind may be plain sync functions or `async def` functions - whichever
    is returned is auto-detected at the point it's called, so real async I/O can be mixed freely
    with plain transforms in the same chain.
    """

    __slots__ = ("_run",)

    def __init__(
        self, run: Callable[[TypeState], Awaitable[Tuple[TypeState, TypeSource]]]
    ) -> None:
        """AsyncState constructor which takes a state-accepting async callable.

        Parameters
        ----------
        run: Callable[[TypeState], Awaitable[Tuple[TypeState, TypeSource]]]
            Callable which, given a state, returns a fresh awaitable each call, resolving to a
            (new_state, value) tuple. To stay re-runnable, `run` must produce a *new* awaitable
            every call rather than handing back an already-created (and possibly already-consumed)
            coroutine object.
        """
        self._run = run

    async def run(self, state: TypeState) -> Tuple[TypeState, TypeSource]:
        """Runs the pipeline with `state` and resolves to the final (new_state, value) tuple.

        Definition: AsyncState(f: s -> (s, a)).run(b) => (b, a)

        Parameters
        ----------
        state: TypeState
            Initial state value to run this AsyncState with.

        Returns
        -------
        result: Tuple[TypeState, TypeSource]
        """
        return await self._run(state)

    @staticmethod
    def from_value(value: TypeSource) -> AsyncState[Any, TypeSource]:
        """Lifts a plain value into an AsyncState which leaves the state unchanged.

        Parameters
        ----------
        value: TypeSource
            Value to be resolved to, alongside the state passed to `run`, unchanged.

        Returns
        -------
        async_state: AsyncState[Any, TypeSource]
        """

        async def run(state: Any) -> Tuple[Any, TypeSource]:
            """Resolves immediately to (state, value), leaving `state` untouched."""
            return state, value

        return AsyncState(run)

    @staticmethod
    def from_state(
        state_monad: State[TypeState, TypeSource],
    ) -> AsyncState[TypeState, TypeSource]:
        """Lifts an existing sync State into an AsyncState.

        Parameters
        ----------
        state_monad: State[TypeState, TypeSource]
            State to be wrapped, run synchronously (unchanged) once `.run(state)` is awaited.

        Returns
        -------
        async_state: AsyncState[TypeState, TypeSource]
        """

        async def run(state: TypeState) -> Tuple[TypeState, TypeSource]:
            """Resolves immediately to `state_monad.run(state)`."""
            return state_monad.run(state)

        return AsyncState(run)

    @staticmethod
    def from_coroutine(
        coroutine_function: Callable[[TypeState], Awaitable[Tuple[TypeState, TypeSource]]],
    ) -> AsyncState[TypeState, TypeSource]:
        """Wraps a state-accepting async callable as an AsyncState.

        Parameters
        ----------
        coroutine_function: Callable[[TypeState], Awaitable[Tuple[TypeState, TypeSource]]]
            Callable which, given a state, returns a fresh awaitable each call (e.g. an
            `async def` function, not an already-created coroutine object, so the AsyncState
            stays re-runnable).

        Returns
        -------
        async_state: AsyncState[TypeState, TypeSource]
        """
        return AsyncState(coroutine_function)

    def map(
        self, function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
    ) -> AsyncState[TypeState, TypeResult]:
        """AsyncState functor interface (>=, map).

        Definition: State(f: s -> (s, a)) >= f: a -> b => State(f: s -> (s, b))

        Parameters
        ----------
        function: Callable[[TypeSource], Union[TypeResult, Awaitable[TypeResult]]]
            Sync or async function to map the value of the state monad and return a new result.

        Returns
        -------
        async_state: AsyncState[TypeState, TypeResult]
            Returns a new AsyncState with the result of the map function and the internal state.
        """

        async def run(state: TypeState) -> Tuple[TypeState, TypeResult]:
            """Runs self with `state`, then maps `function` over the resulting value."""
            new_state, result = await self.run(state)
            return new_state, await _resolve(function(result))

        return AsyncState(run)

    def bind(
        self,
        function: Callable[
            [TypeSource],
            Union[
                AsyncState[TypeState, TypeResult],
                State[TypeState, TypeResult],
                Awaitable[Tuple[TypeState, TypeResult]],
            ],
        ],
    ) -> AsyncState[TypeState, TypeResult]:
        """AsyncState bind interface (>>=, bind, flatMap).

        Definition: State(f: s -> (s, a)) >>= f: a -> State(s, b) => State(f: s -> (s, b))

        Parameters
        ----------
        function: Callable[[TypeSource], AsyncState[TypeState, TypeResult] | State[TypeState, TypeResult] | Awaitable[Tuple[TypeState, TypeResult]]]
            Function which binds the value of the current state monad to a new state monad with a
            new value type - returning another AsyncState, a plain State, or an awaitable
            resolving to a (new_state, value) tuple. Whichever shape is returned is auto-detected.

        Returns
        -------
        async_state: AsyncState[TypeState, TypeResult]
            Returns a new AsyncState with the result of the bind function and the internal state.
        """

        async def run(state: TypeState) -> Tuple[TypeState, TypeResult]:
            """Runs self with `state`, then binds `function` to the resulting value and new state."""
            new_state, value = await self.run(state)
            result = function(value)
            if isinstance(result, AsyncState):
                return await result.run(new_state)
            if isinstance(result, State):
                return result.run(new_state)
            return await _resolve(result)

        return AsyncState(run)

    def apply(
        self, applicative: AsyncState[TypeState, Callable[..., TypeResult]]
    ) -> AsyncState[TypeState, TypeResult]:
        """AsyncState applicative interface for AsyncStates containing a value (<*>).

        Definition: State(f: s -> a) <*> State(f: s -> f: a -> b) => State(f: s -> b)

        Parameters
        ----------
        applicative: AsyncState[TypeState, Callable[[TypeSource], TypeResult]]
            Applicative AsyncState which contains a function and will be applied to the
            AsyncState containing a value.

        Returns
        -------
        async_state: AsyncState[TypeState, TypeResult]
            Applies an AsyncState containing a value to an AsyncState containing a function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncState[TypeState, TypeResult]:
            """Maps the applicative's function, curried, over this AsyncState's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: AsyncState[TypeState, Callable[..., TypeResult]],
        applicative_value: AsyncState[TypeState, Any],
    ) -> AsyncState[TypeState, TypeResult]:
        """AsyncState applicative interface for AsyncStates containing a function (<*>).

        Definition: State(f: e -> f: a -> b) <*> State(f: e -> a) => State(f: e -> b)

        Parameters
        ----------
        applicative_value: AsyncState[TypeState, TypeSource]
            AsyncState value which will be applied to the AsyncState containing a function.

        Returns
        -------
        async_state: AsyncState[TypeState, TypeResult]
            Applies an AsyncState containing a function to an AsyncState with a value or function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> AsyncState[TypeState, TypeResult]:
            """Maps the applicative value's function, curried, over this AsyncState's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    @classmethod
    def get(
        cls: Type[AsyncState[TypeState, TypeSource]],
    ) -> AsyncState[TypeState, TypeState]:
        """AsyncState specific get function.

        Returns
        -------
        async_state: AsyncState[TypeState, TypeState]
            Returns the state from the internals of the monad once run.
        """

        async def run(state: TypeState) -> Tuple[TypeState, TypeState]:
            """Resolves immediately to (state, state)."""
            return state, state

        return AsyncState(run)

    @classmethod
    def put(
        cls: Type[AsyncState[TypeState, TypeSource]], new_state: TypeState
    ) -> AsyncState[TypeState, Tuple[TypeState, Any]]:
        """AsyncState specific put function.

        Returns
        -------
        async_state: AsyncState[TypeState, TypeState]
            Replaces the state inside the monad, and the value with an empty tuple, once run.
        """

        async def run(_: TypeState) -> Tuple[TypeState, Any]:
            """Ignores the current state and replaces it with `new_state`."""
            return new_state, ()

        return AsyncState(run)

    def __str__(self) -> str:
        """Returns the string representation of the AsyncState."""
        return f"AsyncState({self._run})"

    def __repr__(self) -> str:
        """Returns the string representation of the AsyncState (same as __str__)."""
        return str(self)
