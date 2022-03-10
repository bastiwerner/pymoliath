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
            Applicative Try Monad which contains a function.

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
            Try Monad value which will be applied to the function within the Try Monad of self.

        Returns
        -------
        try: Try[TypeResult]:
            Returns a new Try Monad (Success/Failure) with the value applied to the internal applicative function.
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
        return f'Right {self._right_value}'


class Left(Either[TypeLeft, Any]):

    def __init__(self, value: TypeLeft):
        self._left_value = value
        self._right_value = Any
        self._is_left = True

    def __str__(self: Left[TypeLeft]) -> str:
        return f'Left {self._left_value}'


TypeSource = TypeVar("TypeSource")


class Try(Generic[TypeSource], abc.ABC):
    """Try monad interface

    Subclasses of try monad should handle exceptions by returning either the success monad or the failure monad.

    Try implementations:
    * Success: represents the correct way which contains the success result
    * Failure: represents the failure way which contains the actual exception
    """
    _success_value: TypeSource  # Private try monad success value which should not be modified
    _failure_value: Exception  # Private try monad failure value which should not be modified
    _is_failure: bool  # Private try monad is failure flag which should not be modified

    def map(self: Try[TypeSource], function: Callable[[TypeSource], TypeResult]) -> Try[TypeResult]:
        """Try Monad functor interface (>=, map).

        The map function will execute the function by checking for any exception.
        It will return either a Try Monad of type Success or of type Failure.

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult.

        Returns
        -------
        try: Try[TypeResult]
            Returns either a Try Monad of type Success with the function result or otherwise
            a Try Monad of type Failure containing the exception.
        """
        try:
            if self._is_failure:
                return Failure(self._failure_value)
            else:
                return Success(function(self._success_value))
        except Exception as e:
            return Failure(e)

    def bind(self: Try[TypeSource], function: Callable[[TypeSource], Try[TypeResult]]) -> Try[TypeResult]:
        """Success try (either) monad bind interface (>>=, bind, flatMap).

        The bind function will execute the passed function and is also checking for any exception.
        It will return either a Try Monad of type Success or of type Failure.

        Parameters
        ----------
        function: Callable[[TypeSource], Try[TypeResult]]
            Function which takes a value of type TypeSource and returns a try monad with a new value type.

        Returns
        -------
        result: Try[TypeResult]
            Returns either a Try Monad of type Success/Failure from the function result or otherwise
            a Try Monad of type Failure containing the exception.
        """
        try:
            if self._is_failure:
                return Failure(self._failure_value)
            else:
                return function(self._success_value)
        except Exception as e:
            return Failure(e)

    def apply(self: Try[TypeSource], applicative: Try[Callable[[TypeSource], TypeResult]]) -> Try[TypeResult]:
        """Try Monad applicative interface for Try Monads containing a value (<*>).

        Parameters
        ----------
        applicative: Try[Callable[Callable[[TypeSource], TypeResult]]
            Applicative Try Monad which contains a function.

        Returns
        -------
        result: Try[TypeResult]
            Returns a new Try Monad (Success/Failure) with the applicative applied.
        """

        def binder(applicative_function: Callable[[TypeSource], TypeResult]) -> Try[TypeResult]:
            def inner(x: TypeSource) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: Try[Callable[[TypePure], TypeResult]], applicative_value: Try[TypePure]) -> Try[TypeResult]:
        """Try Monad applicative interface for Try Monads containing a function (<*>).

        Parameters
        ----------
        applicative_value: Try[TypePure]
            Try Monad value which will be applied to the function within the Try Monad of self.

        Returns
        -------
        try: Try[TypeResult]:
            Returns a new Try Monad (Success/Failure) with the value applied to the internal applicative function.
        """

        def binder(applicative_function: Callable[[TypePure], TypeResult]) -> Try[TypeResult]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def success_or_else(self, default_value: TypeSource) -> TypeSource:
        if self._is_failure:
            return default_value
        else:
            return self._success_value

    def failure_or_else(self, default_value: Exception) -> Exception:
        if self._is_failure:
            return self._failure_value
        else:
            return default_value

    def either(self: Try[TypeSource], failure_function: Callable[[Exception], TypeResult],
               success_function: Callable[[TypeSource], TypeResult]) -> TypeResult:
        """Try Monad specific function to handle railroad orientated types.

        Parameters
        ----------
        failure_function: Callable[[TypeErr], TypeResult]
            Callback function for either monads of type Failure
        success_function: Callable[[TypeSource], TypeResult]
            Callback function for either monads of type Success
        """
        if self._is_failure:
            return failure_function(self._failure_value)
        else:
            return success_function(self._success_value)

    def is_success(self: Try[TypeSource]) -> bool:
        """Try monad is success function

        Returns
        -------
        result: bool
            True: if try monad is of type success, False: if try monad is of type failure
        """
        return not self._is_failure

    def is_failure(self: Try[TypeSource]) -> bool:
        """Try monad is failure function

        Returns
        -------
        result: bool
            True: if try monad is of type failure, False: if try monad is of type success
        """
        return self._is_failure

    @abc.abstractmethod
    def __str__(self: Try[TypeSource]) -> str:
        pass

    def __eq__(self: Try[TypeSource], __o: object) -> bool:
        return str(self) == str(__o)

    def __repr__(self: Try[TypeSource]) -> str:
        return str(self)


class Success(Try[TypeSource]):

    def __init__(self, value: TypeSource):
        self._success_value = value
        self._failure_value = Any
        self._is_failure = False

    def __str__(self: Success[TypeSource]) -> str:
        return f'Success {self._success_value}'


class Failure(Try[Any]):

    def __init__(self, value: Exception):
        assert isinstance(value, Exception), "Failure value must be of type Exception"
        self._success_value = Any
        self._failure_value = value
        self._is_failure = True

    def __str__(self: Failure) -> str:
        return f'Failure {self._failure_value}'
