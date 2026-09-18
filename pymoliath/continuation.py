from __future__ import annotations

from typing import Any, Callable, TypeVar, Generic

from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeReturn = TypeVar("TypeReturn")
TypeResult = TypeVar("TypeResult")


class Continuation(Generic[TypeSource, TypeReturn]):
    """The Continuation monad represents computations in continuation-passing style (CPS).
    In continuation-passing style function result is not returned, but instead is passed to another function,
    received as a parameter (continuation).
    """

    _computation: Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]

    def __init__(
        self, computation: Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]
    ):
        """Continuation Monad constructor which takes a computation of type
        Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn].

        Parameters
        ----------
        computation: Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]
            Callable to be stored in the Continuation Monad, invoked with a callback when `run` is called.
        """
        self._computation = computation

    def map(
        self: Continuation[TypeSource, TypeReturn],
        function: Callable[[TypeSource], TypeResult],
    ) -> Continuation[TypeResult, TypeReturn]:
        """Maps the given function by composing it with the continuation computation.

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Returns a new continuation monad with the composed computation and the passed function.
        """

        def mapping(callback: Callable[[TypeResult], TypeReturn]) -> TypeReturn:
            """Runs this continuation with a callback that first applies `function`, then `callback`."""
            return self.run(lambda source: callback(function(source)))

        return Continuation(mapping)

    def bind(
        self: Continuation[TypeSource, TypeReturn],
        function: Callable[[TypeSource], Continuation[TypeResult, TypeReturn]],
    ) -> Continuation[TypeResult, TypeReturn]:
        """Binds the given function by applying the continuation computation to the resulting continuation
        from passed function.

        Parameters
        ----------
        function: Callable[[TypeSource], Continuation[TypeResult, TypeReturn]]

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Returns a new continuation monad with the function binded to the continuation computation.
        """

        return Continuation(lambda cont: self.run(lambda a: function(a).run(cont)))

    def apply(
        self: Continuation[TypeSource, TypeReturn],
        applicative: Continuation[Callable[..., TypeResult], TypeReturn],
    ) -> Continuation[TypeResult, TypeReturn]:
        """Continuation Monad applicative interface for continuations containing a value (<*>).

        Parameters
        ----------
        applicative: Continuation[Callable[[TypeSource], TypeResult], TypeReturn]
            Applicative continuation which contains a function and will be applied to this continuation.

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Applies a continuation containing a value to a continuation containing a function.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Continuation[TypeResult, TypeReturn]:
            """Maps the applicative's function, curried, over this Continuation's value."""
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Continuation[Callable[..., TypeResult], TypeReturn],
        applicative_value: Continuation[Any, TypeReturn],
    ) -> Continuation[TypeResult, TypeReturn]:
        """Continuation Monad applicative interface for continuations containing a function (<*>).

        Parameters
        ----------
        applicative_value: Continuation[TypeSource, TypeReturn]
            Continuation which contains a value and will be applied to this continuation containing a function.

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Applies a continuation containing a function to a continuation containing a value.
        """

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Continuation[TypeResult, TypeReturn]:
            """Maps the applicative value's function, curried, over this Continuation's function value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def run(self, callback: Callable[[TypeSource], TypeReturn]) -> TypeReturn:
        """Run the computation of the continuation with the passed callback"""
        return self._computation(callback)
