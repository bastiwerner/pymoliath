from __future__ import annotations

from functools import partial
from itertools import chain
from typing import Any, TypeVar, List, Callable

TypeSource = TypeVar('TypeSource')
TypeResult = TypeVar('TypeResult')
TypePure = TypeVar('TypePure')


class ListMonad(List[TypeSource]):

    def map(self: ListMonad[TypeSource], function: Callable[[TypeSource], TypeResult]) -> ListMonad[TypeResult]:
        """ListMonad monad functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        list: ListMonad[TypeResult]
            Returns a new list monad with the elements applied to the function.
        """
        return self.__class__(map(function, self))

    def bind(self: ListMonad[TypeSource], function: Callable[[TypeSource], ListMonad[TypeResult]]) -> ListMonad[
        TypeResult]:
        """ListMonad monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], ListMonad[TypeResult]]
            Function which takes a value of type TypeSource and returns an io monad of type TypeResult.

        Returns
        -------
        list: ListMonad[TypeResult]
            Returns a list monad from the result of the function call.
        """
        return self.__class__(chain.from_iterable(map(function, self)))

    def apply(self: ListMonad[TypeSource], applicative: ListMonad[Callable[[TypeSource], TypeResult]]) -> ListMonad[
        TypeResult]:
        """ListMonad monad applicative interface for list monads containing a function returning a value (<*>).

        Parameters
        ----------
        applicative: ListMonad[Callable[[TypeSource], TypeResult]]
            Applicative list monad which contains a function and will be applied to the list monad containing values.

        Returns
        -------
        list: ListMonad[TypeResult]
            Applies a list monad containing values of type TypeSource to an list monad containing a function
            of type Callable[[TypeSource], TypeResult].
        """

        def binder(applicative_function: Callable[[TypeSource], TypeResult]) -> ListMonad[TypeResult]:
            def inner(x: TypeSource) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: ListMonad[Callable[[TypePure], TypeResult]], applicative_value: ListMonad[TypePure]) -> ListMonad[
        TypeResult]:
        """ListMonad monad applicative interface for list monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: ListMonad[TypePure]
            ListMonad monad value which will be applied to the io monad containing a function

        Returns
        -------
        list: ListMonad[TypeResult]
            Applies an list monad containing a function of type Callable[[TypePure], TypeResult]
            to a list monad of type TypePure (value or function).
        """

        def binder(applicative_function: Callable[[TypePure], TypeResult]) -> ListMonad[TypeResult]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def __str__(self: ListMonad[TypeSource]) -> str:
        return f'ListMonad {super().__str__()}'
