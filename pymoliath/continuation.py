from __future__ import annotations

from typing import Callable, TypeVar, Generic

from pymoliath.util import compose

TypeSource = TypeVar("TypeSource")
TypeReturn = TypeVar("TypeReturn")
TypeResult = TypeVar("TypeResult")


class Continuation(Generic[TypeSource, TypeReturn]):
    """The Either Monad represents values with two possibilities: either Left[A] or Right[B].
    """
    _computation: Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]

    def __init__(self, computation: Callable[[Callable[[TypeSource], TypeReturn]], TypeReturn]):
        self._computation = computation

    def map(self: Continuation[TypeSource, TypeReturn], function: Callable[[TypeSource], TypeResult]) \
            -> Continuation[TypeResult, TypeReturn]:
        """Maps the given function by composing it with the continuation computation.

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]

        Returns
        -------
        continuation: Continuation[TypeResult, TypeReturn]
            Returns a new continuation monad with the composed computation and the passed function.
        """

        # Function signature: Callable[Callable[[TypeSource], TypeReturn], TypeResult]
        def mapping(computation: Callable[[TypeSource], TypeReturn]) -> TypeResult:
            return self.run(compose(computation, function))  # type: ignore 

        return Continuation(mapping)  # type: ignore 

    def bind(self: Continuation[TypeSource, TypeReturn],
             function: Callable[[TypeSource], Continuation[TypeResult, TypeReturn]]) -> Continuation[
        TypeResult, TypeReturn]:
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

    def run(self, callback: Callable[[TypeSource], TypeReturn]) -> TypeReturn:
        """Run the computation of the continuation with the passed callback"""
        return self._computation(callback)
