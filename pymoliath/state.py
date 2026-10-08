"""
# State Monad

The State Monad represents a computation that threads a piece of mutable-looking state through a
series of function calls without any actual mutation. It wraps a
`Callable[[TypeState], Tuple[TypeState, TypeSource]]`: given the current state, it produces a new
state alongside a result value, letting stateful-looking code be written in a purely functional
style, with `bind` handling the plumbing of passing the updated state from one step to the next.

* Haskell: [Control.Monad.State](https://hackage.haskell.org/package/mtl/docs/Control-Monad-State.html)

This implementation is heavily inspired by Haskell's `State` monad.

The `State[TypeState, TypeSource]` type wraps a `Callable[[TypeState], Tuple[TypeState, TypeSource]]`:
a computation which, given the current state, returns the new state and a result value.

```python
State[TypeState, TypeSource]
```

## Practical Examples and Benefits:

The State Monad is particularly useful for simulating mutable state (a counter, an accumulator, a
random-number generator's seed) in a purely functional way, without global variables or explicit
threading of a state parameter through every function signature.

### Benefits:
1. Implicit State Threading: `bind` passes the updated state from one step to the next
   automatically, so intermediate functions don't need an explicit state parameter.
2. No Hidden Mutation: The "state" is just a value passed along and returned, never mutated in
   place, making the data flow explicit and easy to reason about.
3. Composability: `get`/`put` combined with `map`/`bind` let you build up a whole stateful
   computation as a value, only actually run once `run(initial_state)` is called.

#### Example: Threading a counter through several steps without a mutable variable.

```python
# Without State (Imperative, mutable)
counter = 0
counter += 1
first = counter
counter += 1
second = counter

# With State (Functional)
increment = State(lambda n: (n + 1, n + 1))
program = increment.bind(lambda first: increment.map(lambda second: (first, second)))
result_state, (first, second) = program.run(0)
```
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Tuple, Type, TypeVar

from pymoliath.util import curry

TypeState = TypeVar("TypeState")
TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")


class State(Generic[TypeState, TypeSource]):
    """State Monad Implementation

    The state monad is a monad that allows for chaining of a state variable (which may be arbitrarily complex)
    through a series of function calls, to simulate stateful code.

    """

    __slots__ = ("_value",)

    _value: Callable[
        [TypeState], Tuple[TypeState, TypeSource]
    ]  # Private state monad value of type callable

    def __init__(
        self, value: Callable[[TypeState], Tuple[TypeState, TypeSource]]
    ) -> None:
        """State Monad constructor which takes a callable of type Callable[[TypeState], Tuple[TypeState, TypeSource]].

        Parameters
        ----------
        value: Callable[[TypeState], Tuple[TypeState, TypeSource]]
            Callable to be stored in the State Monad, invoked with the current state when `run` is called.

        Examples
        --------
        >>> state: State[int, int] = State(lambda n: (n + 1, n))
        >>> state.run(0)
        (1, 0)
        """
        if not callable(value):
            raise TypeError("State Monad value must be of type Callable")
        self._value = value

    def map(
        self: State[TypeState, TypeSource], function: Callable[[TypeSource], TypeResult]
    ) -> State[TypeState, TypeResult]:
        """State monad functor interface (>=, map)

        Definition: State(f: s -> (s, a)) >= f: a -> b => State(f: s -> (s, b))

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function to map the value of the state monad and return a new result.

        Returns
        -------
        state: State[TypeState, TypeResult]
            Returns a new state monad with the result of the map function and the internal state.

        Examples
        --------
        >>> state: State[int, int] = State(lambda n: (n + 1, n))
        >>> state.map(lambda x: x * 2).run(0)
        (1, 0)
        """

        def mapper(state: TypeState) -> Tuple[TypeState, TypeResult]:
            """Runs the state monad with `state`, then maps `function` over the resulting value."""
            new_state, result = self.run(state)
            return new_state, function(result)

        return State(mapper)

    def bind(
        self: State[TypeState, TypeSource],
        function: Callable[[TypeSource], State[TypeState, TypeResult]],
    ) -> State[TypeState, TypeResult]:
        """State monad bind interface (>>=, bind, flatMap)

        Definition: State(f: s -> (s, a)) >>= f: a -> State(s, b) => State(f: s -> (s, b))

        Parameters
        ----------
        function: Callable[[TypeSource], State[TypeState, TypeResult]
            Function which binds the value of the current state monad to a new state monad with a new value type.

        Returns
        -------
        state: State[TypeState, TypeResult]
            Returns a new state monad with the result of the bind function and the internal state.

        Examples
        --------
        >>> state: State[int, int] = State(lambda n: (n + 1, n))
        >>> state.bind(lambda x: State(lambda n: (n + 1, x + n))).run(0)
        (2, 1)
        """

        def mapper(state: TypeState) -> Tuple[TypeState, TypeResult]:
            """Runs the state monad with `state`, then binds `function` to the resulting value and new state."""
            new_state, value = self.run(state)
            return function(value).run(new_state)

        return State(mapper)

    def apply(
        self: State[TypeState, TypeSource],
        applicative: State[TypeState, Callable[..., TypeResult]],
    ) -> State[TypeState, TypeResult]:
        """State monad applicative interface for state monads containing a value (<*>).

        Definition: State(f: s -> a) <*> State(f: s -> f: a -> b) => State(f: s -> b)

        Parameters
        ----------
        applicative: State[TypeState, Callable[[TypeSource], TypeResult]]
            Applicative state monad which contains a function and will be applied to the state monad containing
            a value.

        Returns
        -------
        state: State[TypeState, TypeResult
            Applies a state monad containing a value to a state monad containing a function.

        Examples
        --------
        >>> val: State[int, int] = State(lambda n: (n, 10))
        >>> func: State[int, Callable[[int], int]] = State(lambda n: (n, lambda x: x * 2))
        >>> val.apply(func).run(0)
        (0, 20)
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> State[TypeState, TypeResult]:
            """Maps the applicative's function, curried, over this State monad's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: State[TypeState, Callable[..., TypeResult]],
        applicative_value: State[TypeState, Any],
    ) -> State[TypeState, TypeResult]:
        """State monad applicative interface for state monads containing a function (<*>).

        Definition: State(f: e -> f: a -> b) <*> State(f: e -> a) => State(f: e -> b)

        Parameters
        ----------
        applicative_value: State[TypeState, TypeSource]
            State monad value which will be applied to the state monad containing a function

        Returns
        -------
        state: State[TypeState, TypeResult
            Applies a state monad containing a function to a state monad with a value or function.

        Examples
        --------
        >>> func: State[int, Callable[[int], int]] = State(lambda n: (n, lambda x: x * 2))
        >>> val: State[int, int] = State(lambda n: (n, 10))
        >>> func.apply2(val).run(0)
        (0, 20)
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> State[TypeState, TypeResult]:
            """Maps the applicative value's function, curried, over this State monad's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    @classmethod
    def get(cls: Type[State[TypeState, TypeSource]]) -> State[TypeState, TypeState]:
        """State monad specific get function.

        Returns the current state as the value, leaving the state itself unchanged.

        Returns
        -------
        state: State[TypeState, TypeState]
            Return the state from the internals of the monad.

        Examples
        --------
        >>> State[int, Any].get().run(10)
        (10, 10)
        """
        return State(lambda state: (state, state))

    @classmethod
    def put(
        cls: Type[State[TypeState, TypeSource]], new_state: TypeState
    ) -> State[TypeState, Tuple[TypeState, Any]]:
        """State monad specific put function.

        Replaces the state inside the monad with `new_state`, and the value with an empty tuple.

        Parameters
        ----------
        new_state: TypeState
            The new state to replace the current state with.

        Returns
        -------
        state: State[TypeState, TypeState]
            Replace the state inside the monad and the value with a tuple.

        Examples
        --------
        >>> State[int, Any].put(42).run(10)
        (42, ())
        """

        def mapper(_: TypeState) -> Tuple[TypeState, Any]:
            """Ignores the current state and replaces it with `new_state`."""
            return new_state, ()

        return State(mapper)

    def run(
        self: State[TypeState, TypeSource], state: TypeState
    ) -> Tuple[TypeState, TypeSource]:
        """State monad lazy run function to start the state monad by passing an initial state value.

        Definition: State(f: s -> (s, a)).run(b) => (b, a)

        Returns
        -------
        result: TypeSource
            Calls the state monad function by passing the state value and returns wrapped result in state and the value.

        Examples
        --------
        >>> state: State[int, int] = State(lambda n: (n + 1, n))
        >>> state.run(0)
        (1, 0)
        """
        new_state, value = self._value(state)
        return new_state, value

    def __str__(self: State[TypeState, TypeSource]) -> str:
        """Returns the string representation of the State Monad.

        Examples
        --------
        >>> str(State(lambda n: (n, n)))  # doctest: +ELLIPSIS
        'State(<function...>)'
        """
        return f"State({self._value})"

    def __repr__(self: State[TypeState, TypeSource]) -> str:
        """Returns the string representation of the State Monad (same as __str__).

        Examples
        --------
        >>> repr(State(lambda n: (n, n)))  # doctest: +ELLIPSIS
        'State(<function...>)'
        """
        return str(self)
