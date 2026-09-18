from __future__ import annotations

from typing import Any, Callable, Generic, Tuple, TypeAlias, TypeVar, cast, overload

from pymoliath.either import Either, Left, Right
from pymoliath.result import Err, Ok, Result
from pymoliath.util import curry

TypeSource = TypeVar("TypeSource")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class Success(Generic[TypeSource]):
    """Try monad interface

    Subclasses of try monad should handle exceptions by returning either the success monad or the failure monad.

    Try implementations:
    * Success: represents the correct way which contains the success result
    * Failure: represents the failure way which contains the actual exception
    """

    __match_args__ = ("_success_value",)

    def __init__(self, value: TypeSource):
        self._success_value = value

    def map(self, function: Callable[[TypeSource], TypeResult]) -> Try[TypeResult]:
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
            return Success(function(self._success_value))
        except Exception as e:
            return Failure(e)

    def map_failure(
        self, function: Callable[[Exception], Exception]
    ) -> Try[TypeSource]:
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
        return self

    def bind(
        self, function: Callable[[TypeSource], Try[TypeResult]]
    ) -> Try[TypeResult]:
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
            return function(self._success_value)
        except Exception as e:
            return Failure(e)

    def bind_failure(
        self, function: Callable[[Exception], Try[TypeSource]]
    ) -> Try[TypeSource]:
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
        return self

    def apply(self, applicative: Try[Callable[..., TypeResult]]) -> Try[TypeResult]:
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

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Try[TypeResult]:
            return self.map(curry(applicative_function))

        return applicative.bind(binder)

    def apply2(
        self: Success[Callable[..., TypeResult]],
        applicative_value: Try[Any],
    ) -> Try[TypeResult]:
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

        def binder(
            applicative_function: Callable[..., TypeResult],
        ) -> Try[TypeResult]:
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def is_success_and(self, function: Callable[[TypeSource], bool]) -> bool:
        """Returns True if the Try Monad is Success and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[TypeSource], bool]
            Predicate function applied to the Success value.

        Returns
        -------
        result: bool
            Returns the predicate result.
        """
        return function(self._success_value)

    def is_failure_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns False, since this Try Monad is Success.

        Parameters
        ----------
        function: Callable[[Exception], bool]
            Predicate function which would be applied to the Failure value.

        Returns
        -------
        result: bool
            Returns False.
        """
        return False

    def map_or(
        self, default_value: TypeResult, function: Callable[[TypeSource], TypeResult]
    ) -> TypeResult:
        """Applies the function to the Success value, or returns the default value if Failure.

        Parameters
        ----------
        default_value: TypeResult
            Default value to be returned if the Try Monad is Failure.
        function: Callable[[TypeSource], TypeResult]
            Function applied to the Success value.

        Returns
        -------
        result: TypeResult
            Returns the function result.
        """
        return function(self._success_value)

    def and_(self, other: Try[TypeResult]) -> Try[TypeResult]:
        """Returns `other` if the Try Monad is Success, otherwise Failure.

        Parameters
        ----------
        other: Try[TypeResult]
            Try Monad to be returned if this Try Monad is Success.

        Returns
        -------
        result: Try[TypeResult]
            Returns `other`.
        """
        return other

    def or_(self, other: Try[TypeSource]) -> Try[TypeSource]:
        """Returns this Try Monad if it is Success, otherwise `other`.

        Parameters
        ----------
        other: Try[TypeSource]
            Try Monad to be returned if this Try Monad is Failure.

        Returns
        -------
        result: Try[TypeSource]
            Returns this Success.
        """
        return self

    def zip(self, other: Try[TypePure]) -> Try[Tuple[TypeSource, TypePure]]:
        """Combines this Try Monad with another into a Try Monad of a tuple, or Failure if either is Failure.

        Parameters
        ----------
        other: Try[TypePure]
            Try Monad to be zipped with this Try Monad.

        Returns
        -------
        result: Try[Tuple[TypeSource, TypePure]]
            Returns Success of a tuple of both values, or Failure.
        """
        return other.map(lambda o: (self._success_value, o))

    @overload
    def flatten(self: Success[Success[TypeResult]]) -> Try[TypeResult]: ...

    @overload
    def flatten(self: Success[Failure]) -> Try[Any]: ...

    def flatten(self) -> Try[Any]:
        """Flattens a nested Try Monad by one level.

        Returns
        -------
        result: Try[TypeResult]
            Returns the nested Try Monad.
        """
        return cast(Try[Any], self._success_value)

    def unwrap(self) -> TypeSource:
        """Returns the Success value if not Failure, or otherwise raises the Failure Exception.

        Returns
        -------
        result: TypeRight
            Returns the Success value or raises the Failure Exception.
        """
        return self._success_value

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
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
        return self._success_value

    def unwrap_or_else(
        self, failure_function: Callable[[Exception], TypeSource]
    ) -> TypeSource:
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
        return self._success_value

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
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
        return default_value

    def inspect(self, function: Callable[[TypeSource], None]) -> Try[TypeSource]:
        """Inspect the Try monad value of TypeSource

        Parameters
        ----------
        function: Callable[[TypeSource], None]
            Inspection function which takes the success value of the Try monad

        Returns
        -------
        try: Try[TypeSource]
        """
        function(self._success_value)
        return self

    def inspect_failure(self, function: Callable[[Exception], None]) -> Try[TypeSource]:
        """Inspect the Try monad Exception value

        Parameters
        ----------
        function: Callable[[Exception], None]
            Inspection function which takes the exception value of the Try monad

        Returns
        -------
        try: Try[TypeSource]
        """
        return self

    def match(
        self,
        failure_function: Callable[[Exception], TypeResult],
        success_function: Callable[[TypeSource], TypeResult],
    ) -> TypeResult:
        """Try Monad specific function to handle railroad orientated types.

        Parameters
        ----------
        failure_function: Callable[[TypeErr], TypeResult]
            Callback function for either monads of type Failure
        success_function: Callable[[TypeSource], TypeResult]
            Callback function for either monads of type Success
        """
        return success_function(self._success_value)

    def to_either(self) -> Either[Exception, TypeSource]:
        """Try Monad specific function to return an Either Monad.

        Returns
        -------
        either: Either[Exception, TypeSource]
            Returns the Try Monad as Either Monad
        """
        return Right(self._success_value)

    def to_result(self) -> Result[TypeSource, Exception]:
        """Try Monad specific function to return an Result Monad.

        Returns
        -------
        either: Either[Exception, TypeSource]
            Returns the Try Monad as Either Monad
        """
        return Ok(self._success_value)

    def is_success(self) -> bool:
        """Try monad is success function

        Returns
        -------
        result: bool
            True: if try monad is of type success, False: if try monad is of type failure
        """
        return True

    def is_failure(self) -> bool:
        """Try monad is failure function

        Returns
        -------
        result: bool
            True: if try monad is of type failure, False: if try monad is of type success
        """
        return False

    def __str__(self) -> str:
        return f"Success({self._success_value})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Success):
            other_success = cast(Success[Any], other)
            return str(self) == str(other_success) and type(
                self._success_value
            ) is type(other_success._success_value)
        return False

    def __repr__(self) -> str:
        return str(self)


class Failure:
    __match_args__ = ("_failure_value",)

    def __init__(self, value: Exception):
        assert isinstance(value, Exception), "Failure value must be of type Exception"
        self._failure_value = value

    def map(self, function: Callable[[Any], TypeResult]) -> Try[TypeResult]:
        return self

    def map_failure(self, function: Callable[[Exception], Exception]) -> Try[Any]:
        try:
            return Failure(function(self._failure_value))
        except Exception as e:
            return Failure(e)

    def bind(self, function: Callable[[Any], Try[TypeResult]]) -> Try[TypeResult]:
        return self

    def bind_failure(
        self, function: Callable[[Exception], Try[TypeSource]]
    ) -> Try[TypeSource]:
        try:
            return function(self._failure_value)
        except Exception as e:
            return Failure(e)

    def apply(self, applicative: Try[Callable[..., TypeResult]]) -> Try[TypeResult]:
        return self

    def apply2(self, applicative_value: Try[Any]) -> Try[Any]:
        return self

    def is_success_and(self, function: Callable[[Any], bool]) -> bool:
        return False

    def is_failure_and(self, function: Callable[[Exception], bool]) -> bool:
        return function(self._failure_value)

    def map_or(
        self, default_value: TypeResult, function: Callable[[Any], TypeResult]
    ) -> TypeResult:
        return default_value

    def and_(self, other: Try[Any]) -> Try[Any]:
        return self

    def or_(self, other: Try[TypeSource]) -> Try[TypeSource]:
        return other

    def zip(self, other: Try[Any]) -> Try[Any]:
        return self

    def flatten(self) -> Try[Any]:
        return self

    def unwrap(self) -> Any:
        raise self._failure_value

    def unwrap_or(self, default_value: TypeSource) -> TypeSource:
        return default_value

    def unwrap_or_else(
        self, failure_function: Callable[[Exception], TypeSource]
    ) -> TypeSource:
        return failure_function(self._failure_value)

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        return self._failure_value

    def inspect(self, function: Callable[[Any], None]) -> Try[Any]:
        return self

    def inspect_failure(self, function: Callable[[Exception], None]) -> Try[Any]:
        function(self._failure_value)
        return self

    def match(
        self,
        failure_function: Callable[[Exception], TypeResult],
        success_function: Callable[[Any], TypeResult],
    ) -> TypeResult:
        return failure_function(self._failure_value)

    def to_either(self) -> Either[Exception, Any]:
        return Left(self._failure_value)

    def to_result(self) -> Result[Any, Exception]:
        return Err(self._failure_value)

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"Failure({self._failure_value})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Failure):
            return str(self) == str(other) and type(self._failure_value) is type(
                other._failure_value
            )
        return False

    def __repr__(self) -> str:
        return str(self)


Try: TypeAlias = Success[TypeSource] | Failure


def safe(function: Callable[[], TypeResult]) -> Try[TypeResult]:
    """Either Monad method (try_except) which wraps a function that may raise an exception.

    Parameters
    ----------
    function: Callable[[], TypeResult]
        Callable function which may raise an exception

    Returns
    -------
    either: Either[Exception, TypeResult]
        Returns an Either Monad which contains either the function result or an Exception with a message added.
    """
    try:
        return Success(function())
    except Exception as e:
        return Failure(e)
