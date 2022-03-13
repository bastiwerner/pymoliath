from __future__ import annotations

from functools import partial
from typing import Any, TypeVar, Generic, Callable, Tuple, Type

TypeState = TypeVar('TypeState')
TypeSource = TypeVar('TypeSource')
TypeResult = TypeVar('TypeResult')


class State(Generic[TypeState, TypeSource]):
    """State Monad Implementation

    The state monad is a monad that allows for chaining of a state variable (which may be arbitrarily complex)
    through a series of function calls, to simulate stateful code.

    """
    _value: Callable[[TypeState], Tuple[TypeState, TypeSource]]  # Private state monad value of type callable

    def __init__(self, value: Callable[[TypeState], Tuple[TypeState, TypeSource]]) -> None:
        if not isinstance(value, Callable):
            raise TypeError("State Monad value must be of type Callable")
        self._value = value

    def map(self: State[TypeState, TypeSource],
            function: Callable[[TypeSource], TypeResult]) -> State[TypeState, TypeResult]:
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
        """

        def mapper(state: TypeState) -> Tuple[TypeState, TypeResult]:
            new_state, result = self.run(state)
            return new_state, function(result)

        return self.__class__(mapper)

    def bind(self: State[TypeState, TypeSource],
             function: Callable[[TypeSource], State[TypeState, TypeResult]]) -> State[TypeState, TypeResult]:
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
        """

        def mapper(state: TypeState):
            new_state, value = self.run(state)
            return function(value).run(new_state)

        return self.__class__(mapper)

    def apply(self: State[TypeState, TypeSource],
              applicative: State[TypeState, Callable[[TypeSource], TypeResult]]) -> State[TypeState, TypeResult]:
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
        """

        def binder(applicative_function: Callable[[TypeSource], TypeResult]) -> State[TypeState, TypeResult]:
            def inner(x: TypeSource) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: State[TypeState, Callable[[TypeSource], TypeResult]],
               applicative_value: State[TypeState, TypeSource]) -> State[TypeState, TypeResult]:
        """Reader monad applicative interface for state monads containing a function (<*>).

        Definition: State(f: e -> f: a -> b) <*> State(f: e -> a) => State(f: e -> b)

        Parameters
        ----------
        applicative_value: State[TypeState, TypeSource]
            Reader monad value which will be applied to the state monad containing a function

        Returns
        -------
        state: State[TypeState, TypeResult
            Applies a state monad containing a function to a state monad with a value or function.
        """

        def binder(applicative_function: Callable[[TypeSource], TypeResult]) -> State[TypeState, TypeResult]:
            def inner(x: TypeSource) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    @classmethod
    def get(cls: Type[State[TypeState, TypeSource]]) -> State[TypeState, TypeState]:
        """State monad specific get function.

        Returns
        -------
        state: State[TypeState, TypeState]
            Return the state from the internals of the monad.
        """
        return cls(lambda state: (state, state))

    @classmethod
    def put(cls: Type[State[TypeState, TypeSource]], new_state: TypeState) -> State[TypeState, Tuple[TypeState, Any]]:
        """State monad specific get function.

        Returns
        -------
        state: State[TypeState, TypeState]
            Replace the state inside the monad and the value with a tuple.
        """

        def mapper(_: TypeState) -> Tuple[TypeState, Any]:
            return new_state, ()

        return cls(mapper)

    def run(self: State[TypeState, TypeSource], state: TypeState) -> Tuple[TypeState, TypeSource]:
        """State monad lazy run function to start the state monad by passing an initial state value.

        Definition: State(f: s -> (s, a)).run(b) => (b, a)

        Returns
        -------
        result: TypeSource
            Calls the state monad function by passing the state value and returns wrapped result in state and the value.
        """
        new_state, value = self._value(state)
        return new_state, value

    def __str__(self: State[TypeState, TypeSource]) -> str:
        return f'State({self._value})'

    def __repr__(self: State[TypeState, TypeSource]) -> str:
        return str(self)
