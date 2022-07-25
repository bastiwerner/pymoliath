from __future__ import annotations

import abc
from functools import partial
from typing import TypeVar, Generic, Callable, Any

from pymoliath.either import Either, Left, Right, Result, Err, Ok

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


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
        """Calls function to the Success value if not an Failure, otherwise leaving the Failure value untouched.
        The map function will execute the function by checking for any exception.

        Parameters
        ----------
        function: Callable[[TypeSource], TypeResult]
            Function which takes a value of TypeSource and returns a value of type TypeResult.

        Returns
        -------
        try: Try[TypeResult]
            Returns an Success with the function result or otherwise Failure.
        """
        try:
            if self._is_failure:
                return Failure(self._failure_value)
            return Success(function(self._success_value))
        except Exception as e:
            return Failure(e)

    def map_failure(self: Try[TypeSource], function: Callable[[Exception], Exception]) -> Try[TypeSource]:
        """Calls function to the Failure value if not Success, otherwise leaving the Success value untouched.
        The map function will execute the function by checking for any exception.

        Parameters
        ----------
        function: Callable[[Exception], Exception]
            Function which takes an Exception and returns an Exception.

        Returns
        -------
        result: Try[TypeSource]
            Returns an Failure with the function result or otherwise Success.
        """
        try:
            if self._is_failure:
                return Failure(function(self._failure_value))
            return Success(self._success_value)
        except Exception as e:
            return Failure(e)

    def bind(self: Try[TypeSource], function: Callable[[TypeSource], Try[TypeResult]]) -> Try[TypeResult]:
        """Calls function if Try Monad is Success, otherwise returns Failure.
        The bind function will execute the passed function and is also checking for any exception.

        Parameters
        ----------
        function: Callable[[TypeSource], Try[TypeResult]]
            Function which takes a value of TypeSource and returns a Try Monad of TypeResult.

        Returns
        -------
        result: Try[TypeResult]
            Returns a Try Monad from the function result if Success, otherwise an Failure.
        """
        try:
            if self._is_failure:
                return Failure(self._failure_value)
            return function(self._success_value)
        except Exception as e:
            return Failure(e)

    def bind_failure(self: Try[TypeSource], function: Callable[[Exception], Try[TypeSource]]) -> Try[TypeSource]:
        """Calls function if Try Monad is an Failure, otherwise returns Success.
        The bind function will execute the passed function and is also checking for any exception.

        Parameters
        ----------
        function: Callable[[Exception], Try[TypeResult]]
            Function which takes an Exception and returns a Try Monad of TypeResult.

        Returns
        -------
        result: Try[TypeResult]
            Returns a Try Monad from the function result if Failure, otherwise an Success.
        """
        try:
            if self._is_failure:
                return function(self._failure_value)
            return Success(self._success_value)
        except Exception as e:
            return Failure(e)

    def apply(self: Try[TypeSource], applicative: Try[Callable[[TypeSource], TypeResult]]) -> Try[TypeResult]:
        """Applies the passed applicative wrapping a function if Try Monad is Success, otherwise returns Failure.

        Parameters
        ----------
        applicative: Try[Callable[Callable[[TypeSource], TypeResult]]
            Applicative Try Monad which contains a function.

        Returns
        -------
        result: Try[TypeResult]
            Returns a Try Monad from the applied function if Success, otherwise an Failure.
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
        """Applies the passed Try Monad wrapping a value to the Try Monad containing a function if Success,
        otherwise returns Failure.

        Parameters
        ----------
        applicative_value: Try[TypePure]
            Try monad containing a value which will be applied to the Try Monad containing a function.

        Returns
        -------
        try: Try[TypeResult]:
            Returns a Try Monad from the applied function if Success, otherwise an Failure.
        """

        def binder(applicative_function: Callable[[TypePure], TypeResult]) -> Try[TypeResult]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def unwrap(self: Try[TypeSource]) -> TypeSource:
        """Returns the Success value if not Failure, or otherwise raises the Failure Exception.

        Returns
        -------
        result: TypeRight
            Returns the Success value or raises the Failure Exception.
        """
        if self._is_failure:
            raise self._failure_value
        return self._success_value

    def unwrap_or(self: Try[TypeSource], default_value: TypeSource) -> TypeSource:
        """Returns the Ok value if not Err, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: TypeOk
            Default value of TypeOk

        Returns
        -------
        result: TypeOk
            Returns the Ok value or a default value.
        """
        if self._is_failure:
            return default_value
        return self._success_value

    def unwrap_or_else(self: Try[TypeSource], failure_function: Callable[[Exception], TypeSource]) -> TypeSource:
        """Returns the Success value if not Failure, or otherwise calls the provided function with the failure value.

        Parameters
        ----------
        failure_function: Callable[[Exception], TypeRight]
            Called with the failure value and must return a value of type TypeSource

        Returns
        -------
        result: TypeRight
            Returns the Success value or value from the function call.
        """
        if self._is_failure:
            return failure_function(self._failure_value)
        return self._success_value

    def unwrap_failure_or(self: Try[TypeSource], default_value: Exception) -> Exception:
        """Returns the Err value if not Ok, or otherwise a provided default Exception.

        Parameters
        ----------
        default_value: Exception
            Default value of type Exception

        Returns
        -------
        result: Exception
            Returns the Err value or a default value.
        """
        if self._is_failure:
            return self._failure_value
        return default_value

    def inspect(self: Try[TypeSource], function: Callable[[TypeSource], None]) -> Try[TypeSource]:
        """Inspect the Try monad value of TypeSource

        Parameters
        ----------
        function: Callable[[TypeSource], None]
            Inspection function which takes the success value of the Try monad

        Returns
        -------
        try: Try[TypeSource]
        """
        if self.is_success():
            function(self._success_value)
        return self

    def inspect_failure(self: Try[TypeSource], function: Callable[[Exception], None]) -> Try[TypeSource]:
        """Inspect the Try monad Exception value

        Parameters
        ----------
        function: Callable[[Exception], None]
            Inspection function which takes the exception value of the Try monad

        Returns
        -------
        try: Try[TypeSource]
        """
        if self.is_failure():
            function(self._failure_value)
        return self

    def match(self: Try[TypeSource], failure_function: Callable[[Exception], TypeResult],
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
        return success_function(self._success_value)

    def to_either(self: Try[TypeSource]) -> Either[Exception, TypeSource]:
        """Try Monad specific function to return an Either Monad.

        Returns
        -------
        either: Either[Exception, TypeSource]
            Returns the Try Monad as Either Monad
        """
        if self._is_failure:
            return Left(self._failure_value)
        return Right(self._success_value)

    def to_result(self: Try[TypeSource]) -> Result[TypeSource, Exception]:
        """Try Monad specific function to return an Result Monad.

        Returns
        -------
        either: Either[Exception, TypeSource]
            Returns the Try Monad as Either Monad
        """
        if self._is_failure:
            return Err(self._failure_value)
        return Ok(self._success_value)

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

    @classmethod
    def safe(cls, function: Callable[[], TypeResult]) -> Try[TypeResult]:
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
            return Success(function())
        except Exception as e:
            return Failure(e)

    @abc.abstractmethod
    def __str__(self: Try[TypeSource]) -> str:
        pass

    def __eq__(self: Try[TypeSource], other: object) -> bool:
        if isinstance(other, Failure):
            return str(self) == str(other) and type(self._failure_value) == type(other._failure_value)
        elif isinstance(other, Success):
            return str(self) == str(other) and type(self._success_value) == type(other._success_value)  # type: ignore
        else:
            return False

    def __repr__(self: Try[TypeSource]) -> str:
        return str(self)


class Success(Try[TypeSource]):

    def __init__(self, value: TypeSource):
        self._success_value = value
        self._failure_value = Any
        self._is_failure = False

    def __str__(self: Success[TypeSource]) -> str:
        return f'Success({self._success_value})'


class Failure(Try[Any]):

    def __init__(self, value: Exception):
        assert isinstance(value, Exception), "Failure value must be of type Exception"
        self._success_value = Any
        self._failure_value = value
        self._is_failure = True

    def __str__(self: Failure) -> str:
        return f'Failure({self._failure_value})'
