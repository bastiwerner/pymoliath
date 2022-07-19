from __future__ import annotations

from functools import partial
from typing import Callable, Generic, TypeVar, Union, Any, Iterable

from pymoliath.list import ListMonad

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


class Sequence(Generic[TypeSource]):
    """Lazy sequence evaluation using the list monad
    """

    def __init__(self, value: Union[Iterable[TypeSource], Callable[[], Iterable[TypeSource]]]):
        """Lazy Sequence Monad constructor/unit function

        Parameters
        ----------
        value: Union[Iterable[TypeSource], Callable[[], Iterable[TypeSource]]]
            Single value or type TypeSource or callable returning a value of type TypeSource
        """
        if isinstance(value, Callable):
            self._callable = lambda: ListMonad(value())
        else:
            self._callable = lambda: ListMonad(value)

    def map(self: Sequence[TypeSource], function: Callable[[TypeSource], TypeResult]) -> Sequence[TypeResult]:
        """Sequence Monad map function

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function to be evaluated lazy

        Returns
        -------
        sequence: Sequence[TypeResult]
            Returns a new sequence monad containing the resulting value
        """
        return self.__class__(lambda: self._callable().map(function))

    def bind(self: Sequence[TypeSource], function: Callable[[TypeSource], Sequence[TypeResult]]) -> Sequence[
        TypeResult]:
        """Sequence Monad bind function

        Parameters
        ----------
        function: Callable[[TypeSource], Sequence[TypeResult]]
            Function to be evaluated lazy

        Returns
        -------
        sequence: Sequence[TypeResult]
            Returns the new sequence monad from the bind function
        """
        return self.__class__(lambda: self._callable().bind(lambda value: ListMonad(function(value).run())))

    def filter(self: Sequence[TypeSource], filter_function: Callable[[TypeSource], bool]) -> Sequence[TypeSource]:
        """Sequence filter function

        Parameters
        ----------
        filter_function: Callable[[TypeSource], bool]
            Filter function for the list

        Returns
        -------
        sequence: Sequence[TypeSource]
            Returns a filtered sequence monad
        """
        return self.__class__(lambda: self._callable().filter(filter_function))

    def take(self: Sequence[TypeSource], amount: int) -> Sequence[TypeSource]:
        """Sequence monad take function

        Parameters
        ----------
        amount: int
            Amount of values to be taken from the list for the next operation.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Takes our only an specific amount of values from the list for further execution.
        """
        return self.__class__(lambda: self._callable().take(amount))

    def skip(self: Sequence[TypeSource], amount: int) -> Sequence[TypeSource]:
        """Sequence monad skip function

        Parameters
        ----------
        amount: int
            Amount of values to be skipped from the list for the next operation.

        Returns
        -------
        sequence: Sequence[TypeSource]
            Skips an amount of values from the list for further execution.
        """
        return self.__class__(lambda: self._callable().skip(amount))

    def apply(self: Sequence[TypeSource], applicative: Sequence[Callable[[TypeSource], TypeResult]]) -> Sequence[
        TypeResult]:
        """Sequence monad applicative interface for sequence monads containing a function returning a value (<*>).

        Parameters
        ----------
        applicative: Sequence[Callable[[TypeSource], TypeResult]]
            Applicative list monad which contains a function and will be applied to the list monad containing values.

        Returns
        -------
        sequence: Sequence[TypeResult]
            Applies a sequnece monad containing values of type TypeSource to an sequence monad containing a function
            of type Callable[[TypeSource], TypeResult].
        """
        return self.__class__(lambda: self._callable().apply(ListMonad(applicative.run())))  # type: ignore

    def apply2(self: Sequence[Callable[[TypePure], TypeResult]], applicative_value: Sequence[TypePure]) -> Sequence[
        TypeResult]:
        """Sequence monad applicative interface for sequence monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: Sequence[TypePure]
            Sequence monad value which will be applied to the sequence monad containing a function

        Returns
        -------
        sequence: Sequence[TypeResult]
            Applies an sequence monad containing a function of type Callable[[TypePure], TypeResult]
            to a sequence monad of type TypePure (value or function).
        """
        return self.__class__(lambda: self._callable().apply2(ListMonad(applicative_value.run())))  # type: ignore

    def run(self) -> Iterable[TypeSource]:
        """Sequence monad lazy evaluation function

        Returns
        -------
        iterable: Iterable[TypeSource]
            Returns an iterable of the lazy evaluated result
        """
        return list(self._callable())

    def __str__(self) -> str:
        return f'Sequence({self._callable})'

    def __repr__(self) -> str:
        return str(self)
