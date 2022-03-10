from __future__ import annotations

from functools import partial
from typing import Any, Callable, Generic, TypeVar

TypeSource = TypeVar('TypeSource')
TypeResult = TypeVar('TypeResult')
TypePure = TypeVar('TypePure')


class IO(Generic[TypeSource]):
    """IO monad implementation

    A value of type IO a is a computation which, when performed, does some I/O before returning a value of type a.
    """
    _value: Callable[[], TypeSource]  # Private io monad value of type callable which should not be modified

    def __init__(self, value: Callable[[], TypeSource]):
        if not isinstance(value, Callable):
            raise TypeError("IO value must be of type Callable")
        self._value = value

    def map(self: IO[TypeSource], function: Callable[[TypeSource], TypeResult]) -> IO[TypeResult]:
        """IO monad functor interface (>=, map).

        Definition: IO(f: _ -> a) >= f: a -> b => IO(f: _ -> b)

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult

        Returns
        -------
        io: IO[TypeResult]
            Returns a new io monad containing the result of the passed function and the io call.
        """
        return self.__class__(lambda: function(self.run()))

    def bind(self: IO[TypeSource], function: Callable[[TypeSource], IO[TypeResult]]) -> IO[TypeResult]:
        """IO monad bind interface (>>=, bind, flatMap).

        Definition: IO(f: _ -> a) >>= f: a -> IO(f: _ -> b) => IO(f: _ -> b)

        Parameters
        ----------
        function: Callable[[TypeSource], IO[TypeResult]]
            Function which takes a value of type TypeSource and returns an io monad of type TypeResult.

        Returns
        -------
        io: IO[TypeResult]
            Returns an io monad with the function call result
        """
        return function(self.run())

    def apply(self: IO[TypeSource], applicative: IO[Callable[[TypeSource], TypeResult]]) -> IO[TypeResult]:
        """IO monad applicative interface for io monads containing a function returning a value (<*>).

        Definition: IO(f: _ -> a) <*> IO(f: _ -> f: a -> b) => IO(f: _ -> b)

        Parameters
        ----------
        applicative: IO[Callable[[TypeSource], TypeResult]]
            Applicative io monad which contains a function and will be applied to the io monad containing a value.

        Returns
        -------
        io: IO[TypeResult]
            Applies an io monad containing a value of type TypeSource to an io monad containing a function
            of type Callable[[TypeSource], TypeResult].
        """

        def binder(applicative_function: Callable[[TypeSource], TypeResult]) -> IO[TypeResult]:
            def inner(x: TypeSource) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: IO[Callable[[TypePure], TypeResult]], applicative_value: IO[TypePure]) -> IO[TypeResult]:
        """IO monad applicative interface for io monads containing a function (<*>).

        Definition: IO(f: _ -> f: a -> b) <*> IO(f: _ -> a) => IO(f: _ -> b)

        Parameters
        ----------
        applicative_value: IO[TypePure]
            IO monad value which will be applied to the io monad containing a function

        Returns
        -------
        io: IO[TypeResult]
            Applies an io monad containing a function of type Callable[[TypePure], TypeResult]
            to an io monad of type TypePure (value or function).
        """

        def binder(applicative_function: Callable[[TypePure], TypeResult]) -> IO[TypeResult]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def run(self: IO[TypeSource]) -> TypeSource:
        """IO monad lazy run function to start the io action.

        Definition: IO(f: _ -> a).run() => a

        Returns
        -------
        result: TypeSource
            Calls the io monad function which returns a value of type TypeSource.
        """
        return self._value()

    def __str__(self) -> str:
        return f'IO {self._value}'

    def __repr__(self) -> str:
        return str(self)
