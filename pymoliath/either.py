from __future__ import annotations

import abc
from functools import partial
from typing import Callable, Any, Generic, TypeVar

TypeLeft = TypeVar("TypeLeft")
TypeRight = TypeVar("TypeRight")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class Either(Generic[TypeLeft, TypeRight], abc.ABC):
    """Either monad abstract class

    The Either type represents values with two possibilities: value of Either[A, B] is either Left[A] or Right[B].
    The Either type is sometimes used to represent a value which is either correct or an error.

    Instances:
    * Left
    * Right
    """
    _left_value: TypeLeft  # Private either monad left value which should not be modified
    _right_value: TypeRight  # Private maybe monad right value which should not be modified
    _is_left: bool  # Private maybe monad is nothing flag which should not be modified

    def map(self: Either[TypeLeft, TypeRight],
            function: Callable[[TypeRight], TypeResult]) -> Either[TypeLeft, TypeResult]:
        """Either Monad functor interface (>=, map).

        Parameters
        ----------
        function: Callable[[TypeRight], TypeResult]
            Function which takes a value of TypeRight and returns a value of type TypeResult

        Returns
        -------
        either: Either[TypeLeft, TypeRight]
            Returns either an Either Monad of type Right with the function result as value
            or an Either Monad of type Left path containing the left value.
        """
        if self._is_left:
            return Left(self._left_value)
        else:
            return Right(function(self._right_value))

    def bind(self: Either[TypeLeft, TypeRight],
             function: Callable[[TypeRight], Either[TypeLeft, TypeResult]]) -> Either[TypeLeft, TypeResult]:
        """Either Monad bind interface (>>=, bind, flatMap).

        Parameters
        ----------
        function: Callable[[TypeRight], Either[TypeLeft, TypeResult]]
            Function which takes a value of type TypeRight and returns an Either Monad with a new value type.

        Returns
        -------
        either: Either[TypeLeft, TypeResult]
            Returns either an Either Monad of type Right from the function result
            or an Either Monad of type Left path containing the left value.
        """
        if self._is_left:
            return Left(self._left_value)
        else:
            return function(self._right_value)

    def apply(self: Either[TypeLeft, TypeRight],
              applicative: Either[TypeLeft, Callable[[TypeRight], TypeResult]]) -> Either[TypeLeft, TypeResult]:
        """Either Monad applicative interface for Either Monads containing a value (<*>).

        Parameters
        ----------
        applicative: Either[TypeLeft, Callable[Callable[[TypeSource], TypeResult]]
            Applicative Result Monad which contains a function.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns the applicative result
        """

        def binder(applicative_function: Callable[[TypeRight], TypeResult]) -> Either[TypeLeft, TypeResult]:
            def inner(x: TypeRight) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: Either[TypeLeft, Callable[[TypePure], TypeResult]],
               applicative_value: Either[TypeLeft, TypePure]) -> Either[TypeLeft, TypeResult]:
        """Either Monad applicative interface for Either Monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: Either[TypeLeft, TypePure]
            Result Monad value which will be applied to the function within the Result Monad of self.

        Returns
        -------
        try: Result[TypeResult]:
            Returns a new Result Monad (Ok/Err) with the value applied to the internal applicative function.
        """

        def binder(applicative_function: Callable[[TypePure], TypeResult]) -> Either[TypeLeft, TypeResult]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def left_or_else(self: Either[TypeLeft, TypeRight], default_value: TypeLeft) -> TypeLeft:
        """Either Monad extraction function (left_or_else)

        Parameters
        ----------
        default_value: TypeLeft
            Default value to be returned if the Either Monad is not of type Left

        Returns
        -------
        result: TypeLeft
            Returns either the left value or a default value.
        """
        if self._is_left:
            return self._left_value
        else:
            return default_value

    def right_or_else(self: Either[TypeLeft, TypeRight], default_value: TypeRight) -> TypeRight:
        """Either Monad extraction function (right_or_else)

        Parameters
        ----------
        default_value: TypeRight
            Default value to be returned if the Either Monad is not of type Right

        Returns
        -------
        result: TypeRight
            Returns either the right value or a default value.
        """
        if self._is_left:
            return default_value
        else:
            return self._right_value

    def either(self: Either[TypeLeft, TypeRight], left_function: Callable[[TypeLeft], TypeResult],
               right_function: Callable[[TypeRight], TypeResult]) -> TypeResult:
        """Right monad specific function to handle railroad orientated types.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], None]
            Callback function for either monads of type Left
        right_function: Callable[[TypeRight], None]
            Callback function for either monads of type Right
        """
        if self._is_left:
            return left_function(self._left_value)
        else:
            return right_function(self._right_value)

    def is_left(self: Either[TypeLeft, TypeRight]) -> bool:
        """Either monad is left function

        Returns
        -------
        result: bool
            True: if either monad is of type left, False: if either monad is of type right
        """
        return self._is_left

    def is_right(self: Either[TypeLeft, TypeRight]) -> bool:
        """Either monad is right function

        Returns
        -------
        result: bool
            True: if either monad is of type right, False: if either monad is of type left
        """
        return not self._is_left

    @classmethod
    def safe(cls, function: Callable[[], TypeResult], msg: str = '') -> Either[Exception, TypeResult]:
        """Either Monad method (try_except) which wraps a function that may raise an exception.

        Parameters
        ----------
        function: Callable[[], TypeResult]
            Callable function which may raise an exception
        msg: str (Optional)
            Message to be added in case of an exception

        Returns
        -------
        either: Either[Exception, TypeResult]
            Returns an Either Monad which contains either the function result or an Exception with a message added.
        """
        try:
            return Right(function())
        except Exception as e:
            return Left(Exception(f'{msg}{e}'))

    @abc.abstractmethod
    def __str__(self: Either[TypeLeft, TypeRight]) -> str:
        pass

    def __eq__(self: Either[TypeLeft, TypeRight], __o: object) -> bool:
        return str(self) == str(__o)

    def __repr__(self: Either[TypeLeft, TypeRight]) -> str:
        return str(self)


class Right(Either[Any, TypeRight]):

    def __init__(self, value: TypeRight) -> None:
        self._left_value = Any
        self._right_value = value
        self._is_left = False

    def __str__(self: Right[TypeRight]) -> str:
        return f'Right({self._right_value})'


class Left(Either[TypeLeft, Any]):

    def __init__(self, value: TypeLeft):
        self._left_value = value
        self._right_value = Any
        self._is_left = True

    def __str__(self: Left[TypeLeft]) -> str:
        return f'Left({self._left_value})'
