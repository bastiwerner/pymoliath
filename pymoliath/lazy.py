from __future__ import annotations

from functools import partial
from typing import Callable, Generic, TypeVar, Union, Any

TypeSource = TypeVar("TypeSource")
TypeRight = TypeVar("TypeRight")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class LazyMonad(Generic[TypeSource]):
    _computation: Callable[[], TypeSource]

    def __init__(self, value: Union[TypeSource, Callable[[], TypeSource]]):
        """Lazy monad constructor which takes either a value of type TypeSource or a callable which
        must return a value of type TypeSource.

        Parameters
        ----------
        value
        """
        if isinstance(value, Callable):
            self._computation = value
        else:
            self._computation = lambda: value

    def map(self: LazyMonad[TypeSource], function: Callable[[TypeSource], TypeResult]) -> LazyMonad[TypeResult]:
        """Lazy monad functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        io: LazyMonad[TypeResult]
            Returns a new lazy monad containing the result of the passed function and the io call.
        """
        return self.__class__(lambda: function(self.run()))

    def bind(self: LazyMonad[TypeSource], function: Callable[[TypeSource], LazyMonad[TypeResult]]) -> LazyMonad[
        TypeResult]:
        """Lazy monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeSource], LazyMonad[TypeResult]]
            Function which takes a value of type TypeSource and returns an io monad of type TypeResult.

        Returns
        -------
        lazy: LazyMonad[TypeResult]
            Returns an lazy monad with the function call result
        """
        return function(self.run())

    def apply(self: LazyMonad[TypeSource], applicative: LazyMonad[Callable[[TypeSource], TypeResult]]) -> LazyMonad[
        TypeResult]:
        """LazyMonad monad applicative interface for lazy monads containing a function returning a value (<*>).

        Parameters
        ----------
        applicative: LazyMonad[Callable[[TypeSource], TypeResult]]
            Applicative io monad which contains a function and will be applied to the io monad containing a value.

        Returns
        -------
        lazy: LazyMonad[TypeResult]
            Applies an lazy monad containing a value of type TypeSource to an lazy monad containing a function
            of type Callable[[TypeSource], TypeResult].
        """

        def binder(applicative_function: Callable[[TypeSource], TypeResult]) -> LazyMonad[TypeResult]:
            def inner(x: TypeSource) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: LazyMonad[Callable[[TypePure], TypeResult]], applicative_value: LazyMonad[TypePure]) -> LazyMonad[
        TypeResult]:
        """LazyMonad monad applicative interface for lazy monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: LazyMonad[TypePure]
            LazyMonad monad value which will be applied to the lazy monad containing a function

        Returns
        -------
        lazy: LazyMonad[TypeResult]
            Applies an lazy monad containing a function of type Callable[[TypePure], TypeResult]
            to an lazy monad of type TypePure (value or function).
        """

        def binder(applicative_function: Callable[[TypePure], TypeResult]) -> LazyMonad[TypeResult]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def run(self) -> TypeSource:
        """Lazy monad lazy run function to start the computation.

        Returns
        -------
        result: TypeSource
            Calls the lazy monad function which returns a value of type TypeSource.
        """
        return self._computation()

    def __str__(self) -> str:
        return f'LazyMonad({self._computation})'

    def __repr__(self) -> str:
        return str(self)
