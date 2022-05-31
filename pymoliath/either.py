from __future__ import annotations

import abc
from functools import partial
from typing import Callable, Any, Generic, TypeVar

TypeLeft = TypeVar("TypeLeft")
TypeRight = TypeVar("TypeRight")
TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")


class Either(Generic[TypeLeft, TypeRight], abc.ABC):
    """The Either Monad represents values with two possibilities: either Left[A] or Right[B].
    """
    _left_value: TypeLeft  # Private either monad left value which should not be modified
    _right_value: TypeRight  # Private maybe monad right value which should not be modified
    _is_left: bool  # Private maybe monad is nothing flag which should not be modified

    def map(self: Either[TypeLeft, TypeRight],
            function: Callable[[TypeRight], TypeResult]) -> Either[TypeLeft, TypeResult]:
        """Calls function to the a wrapped Right value if not Left, otherwise leaving the Left value untouched.

        Parameters
        ----------
        function: Callable[[TypeRight], TypeResult]
            Function which takes a value of TypeRight and returns a value of type TypeResult.

        Returns
        -------
        either: Either[TypeLeft, TypeRight]
            Returns Right with the function result or otherwise Left.
        """
        if self._is_left:
            return Left(self._left_value)
        return Right(function(self._right_value))

    def map_left(self: Either[TypeLeft, TypeRight],
                 function: Callable[[TypeLeft], TypeResult]) -> Either[TypeResult, TypeRight]:
        """Calls function to the a wrapped Left value if not Right, otherwise leaving the Right value untouched.

        Parameters
        ----------
        function: Callable[[TypeRight], TypeResult]
            Function which takes a value of TypeLeft and returns a value of type TypeResult.

        Returns
        -------
        either: Either[TypeResult, TypeRight]
            Returns Left with the function result or otherwise Right.
        """
        if self._is_left:
            return Left(function(self._left_value))
        return Right(self._right_value)

    def bind(self: Either[TypeLeft, TypeRight],
             function: Callable[[TypeRight], Either[TypeLeft, TypeResult]]) -> Either[TypeLeft, TypeResult]:
        """Calls function if Either Monad is Right, otherwise returns Left.

        Parameters
        ----------
        function: Callable[[TypeRight], Either[TypeLeft, TypeResult]]
            Function which takes a value of TypeRight and returns a Result Monad of TypeResult.

        Returns
        -------
        either: Either[TypeLeft, TypeResult]
            Returns an Either Monad from the function result if Right, otherwise Left.
        """
        if self._is_left:
            return Left(self._left_value)
        return function(self._right_value)

    def bind_left(self: Either[TypeLeft, TypeRight],
                  function: Callable[[TypeLeft], Either[TypeResult, TypeRight]]) -> Either[TypeResult, TypeRight]:
        """Calls function if Either Monad is Left, otherwise returns Right.

        Parameters
        ----------
        function: Callable[[TypeLeft], Either[TypeResult, TypeRight]]
            Function which takes a value of TypeLeft and returns a Result Monad of TypeResult.

        Returns
        -------
        either: Either[TypeResult, TypeRight]
            Returns a Either Monad from the function result if Left, otherwise Right.
        """
        if self._is_left:
            return function(self._left_value)
        return Right(self._right_value)

    def apply(self: Either[TypeLeft, TypeRight],
              applicative: Either[TypeLeft, Callable[[TypeRight], TypeResult]]) -> Either[TypeLeft, TypeResult]:
        """Applies the passed applicative wrapping a function if Either Monad is Right, otherwise returns Left.

        Parameters
        ----------
        applicative: Either[TypeLeft, Callable[Callable[[TypeSource], TypeResult]]
            Applicative Either Monad which contains a function.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns an Either Monad from the applied function if Right, otherwise Left.
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
        """Applies the passed Either Monad wrapping a value to the Either Monad containing a function if Right,
        otherwise returns Left.

        Parameters
        ----------
        applicative_value: Either[TypeLeft, TypePure]
            Result Monad which contains a value.

        Returns
        -------
        result: Either[TypeLeft, TypeResult]
            Returns an Either Monad from the applied function if Right, otherwise Left.
        """

        def binder(applicative_function: Callable[[TypePure], TypeResult]) -> Either[TypeLeft, TypeResult]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def unwrap(self: Either[TypeLeft, TypeRight]) -> TypeRight:
        """Returns the Right value if not Left, or otherwise raises an Exception containing the left value.

        Returns
        -------
        result: TypeRight
            Returns the Right value or raises an Exception.
        """
        if self._is_left:
            raise Exception(self._left_value)
        return self._right_value

    def unwrap_or(self: Either[TypeLeft, TypeRight], default_value: TypeRight) -> TypeRight:
        """Returns the Right value if not Left, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: TypeRight
            Default value of TypeRight

        Returns
        -------
        result: TypeRight
            Returns the Right value or a default value.
        """
        if self._is_left:
            return default_value
        return self._right_value

    def unwrap_or_else(self: Either[TypeLeft, TypeRight], left_function: Callable[[TypeLeft], TypeRight]) -> TypeRight:
        """Returns the Right value if not Left, or otherwise a provided function which will be called with the left.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], TypeRight]
            Called with the left value and must return a value of type TypeRight

        Returns
        -------
        result: TypeRight
            Returns the Right value or value from the function call.
        """
        if self._is_left:
            return left_function(self._left_value)
        return self._right_value

    def unwrap_left_or(self: Either[TypeLeft, TypeRight], default_value: TypeLeft) -> TypeLeft:
        """Returns the Left value if not Right, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: TypeLeft
            Default value of TypeLeft

        Returns
        -------
        result: TypeLeft
            Returns the Left value or a default value.
        """
        if self._is_left:
            return self._left_value
        return default_value

    def match(self: Either[TypeLeft, TypeRight], left_function: Callable[[TypeLeft], TypeResult],
              right_function: Callable[[TypeRight], TypeResult]) -> TypeResult:
        """Right monad specific function to handle railroad orientated programming.

        Parameters
        ----------
        left_function: Callable[[TypeLeft], None]
            Callback function for either monads of type Left
        right_function: Callable[[TypeRight], None]
            Callback function for either monads of type Right
        """
        if self._is_left:
            return left_function(self._left_value)
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

    def to_result(self: Either[TypeLeft, TypeRight]) -> Result[TypeRight, TypeLeft]:
        return self.match(lambda l: Err(l),
                          lambda r: Ok(r))

    @classmethod
    def safe(cls, function: Callable[[], TypeResult]) -> Either[Exception, TypeResult]:
        """Calls an unsafe function which might raise an Exception and returns Right with the result, otherwise Left
        containing the Exception.

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
            return Left(e)

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


TypeReturn = TypeVar("TypeReturn")
TypeOk = TypeVar("TypeOk")
TypeErr = TypeVar("TypeErr")


class Result(Generic[TypeOk, TypeErr], abc.ABC):
    """Result is a Monad that represents either success (Ok) or failure (Err).
    """
    _ok_value: TypeOk  # Private Result Monad Success value which should not be modified
    _err_value: TypeErr  # Private Result Monad Err value which should not be modified
    _is_err: bool  # Private Result Monad is Err flag which should not be modified

    def map(self: Result[TypeOk, TypeErr], function: Callable[[TypeOk], TypeReturn]) -> Result[TypeReturn, TypeErr]:
        """Calls function to the a wrapped Ok value if not an Err, otherwise leaving the Err value untouched.
        The map function will execute the function by checking for any exception.

        Parameters
        ----------
        function: Callable[[TypeOk], TypeReturn]
            Function which takes a value of TypeOk and returns a value of type TypeReturn.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]:
            Returns an Ok with the function result or otherwise Err.
        """
        if self._is_err:
            return Err(self._err_value)
        return Ok(function(self._ok_value))

    def map_err(self: Result[TypeOk, TypeErr], function: Callable[[TypeErr], TypeReturn]) -> Result[TypeOk, TypeReturn]:
        """Calls function to the a wrapped Err value if not Ok, otherwise leaving the Ok value untouched.
        The map function will execute the function by checking for any exception.

        Parameters
        ----------
        function: Callable[[TypeErr], TypeReturn]
            Function which takes a value of TypeErr and return a value of TypeErr.

        Returns
        -------
        result: Result[TypeOk, TypeReturn]
            Returns an Err with the function result or otherwise Ok.
        """
        if self._is_err:
            return Err(function(self._err_value))
        return Ok(self._ok_value)

    def bind(self: Result[TypeOk, TypeErr], function: Callable[[TypeOk], Result[TypeReturn, TypeErr]]) -> Result[
        TypeReturn, TypeErr]:
        """Calls function if Result Monad is Ok, otherwise returns Err.
        The bind function will execute the passed function and is also checking for any exception.

        Parameters
        ----------
        function: Callable[[TypeOk], Result[TypeReturn, TypeErr]]
            Function which takes a value of TypeOk and returns a new Result Monad.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns a Result Monad from the function result if Ok, otherwise an Err.
        """
        if self._is_err:
            return Err(self._err_value)
        return function(self._ok_value)

    def bind_err(self: Result[TypeOk, TypeErr], function: Callable[[TypeErr], Result[TypeOk, TypeReturn]]) -> Result[
        TypeOk, TypeReturn]:
        """Calls function if Result Monad is an Err, otherwise returns Ok.
        The bind function will execute the passed function and is also checking for any exception.

        Parameters
        ----------
        function: Callable[[TypeErr], Result[TypeOk, TypeReturn]]
            Function which takes a value of TypeErr and returns a new Result Monad.

        Returns
        -------
        result: Result[TypeOk, TypeReturn]
            Returns a Result Monad from the function result if Err, otherwise an Ok.
        """
        if self._is_err:
            return function(self._err_value)
        return Ok(self._ok_value)

    def apply(self: Result[TypeOk, TypeErr], applicative: Result[Callable[[TypeOk], TypeReturn], TypeErr]) -> Result[
        TypeReturn, TypeErr]:
        """Applies the passed applicative wrapping a function if Result Monad is Ok, otherwise returns Err.

        Parameters
        ----------
        applicative: Result[Callable[[TypeOk], TypeReturn], TypeErr]
            Applicative Result Monad which contains a function.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns a Result Monad from the applied function if Ok, otherwise an Err.
        """

        def binder(applicative_function: Callable[[TypeOk], TypeReturn]) -> Result[TypeReturn, TypeErr]:
            def inner(x: TypeOk) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(self: Result[Callable[[TypePure], TypeReturn], TypeErr], applicative_value: Result[TypePure, TypeErr]) \
            -> Result[TypeReturn, TypeErr]:
        """Applies the passed Result Monad wrapping a value to the Result Monad containing a function if Ok,
        otherwise returns Err.

        Parameters
        ----------
        applicative_value: Result[TypePure, TypeErr]
            Result monad which contains a value.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns a Result Monad from the applied function if Ok, otherwise an Err.
        """

        def binder(applicative_function: Callable[[TypePure], TypeReturn]) -> Result[TypeReturn, TypeErr]:
            def inner(x: TypePure) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return applicative_value.map(inner)

        return self.bind(binder)

    def unwrap(self) -> TypeOk:
        """Returns the Ok value if not Err, or otherwise raises an Exception with the Err value.

        Returns
        -------
        result: TypeOk
            Returns the Ok value or a default value.
        """
        if self._is_err:
            raise Exception(self._err_value)
        return self._ok_value

    def unwrap_or(self, default_value: TypeOk) -> TypeOk:
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
        if self._is_err:
            return default_value
        return self._ok_value

    def unwrap_or_else(self, err_function: Callable[[TypeErr], TypeOk]) -> TypeOk:
        """Returns the Ok value if not Err, or otherwise calls the err_function.

        Parameters
        ----------
        err_function: Callable[[TypeErr], TypeOk]
            Error function which will be called if the result is of type Err.

        Returns
        -------
        result: TypeOk
            Returns the Ok value or a default value.
        """
        if self._is_err:
            return err_function(self._err_value)
        return self._ok_value

    def unwrap_err_or(self, default_value: TypeErr) -> TypeErr:
        """Returns the Err value if not Ok, or otherwise a provided default of TypeErr.

        Parameters
        ----------
        default_value: TypeErr
            Default value of TypeErr

        Returns
        -------
        result: TypeErr
            Returns the Err value or a default value.
        """
        if self._is_err:
            return self._err_value
        return default_value

    def match(self: Result[TypeOk, TypeErr], err_function: Callable[[TypeErr], TypeReturn],
              ok_function: Callable[[TypeOk], TypeReturn]) -> TypeReturn:
        """Matches the Result Monad to either an Err function or an Ok function with the same return type.

        Parameters
        ----------
        err_function: Callable[[Exception], TypeReturn]
            Callback function for either monads of type Err
        ok_function: Callable[[TypeOk], TypeReturn]
            Callback function for either monads of type Ok
        """
        if self._is_err:
            return err_function(self._err_value)
        return ok_function(self._ok_value)

    def to_either(self: Result[TypeOk, TypeErr]) -> Either[TypeErr, TypeOk]:
        """Converts the Result Monad to an either monad.

        Returns
        -------
        either: Either[Exception, TypeOk]
            Returns the Result Monad as Either Monad.
        """
        return self.match(lambda l: Left(l),
                          lambda r: Right(r))

    def is_ok(self: Result[TypeOk, TypeErr]) -> bool:
        """Returns True if the Result Monad is Ok, otherwise False if Err.

        Returns
        -------
        result: bool
            True: if Result Monad is Ok, False: if Result Monad is Err.
        """
        return not self._is_err

    def is_err(self: Result[TypeOk, TypeErr]) -> bool:
        """Try monad is Err function

        Returns
        -------
        result: bool
            True: if Result Monad is Err, False: if Result Monad is Ok.
        """
        return self._is_err

    @classmethod
    def safe(cls, function: Callable[[], TypeReturn]) -> Result[TypeReturn, Exception]:
        """Calls an unsafe function which might raise an Exception and returns Ok with the result, otherwise Err.

        Parameters
        ----------
        function: Callable[[], TypeReturn]
            Callable function which may raise an exception.

        Returns
        -------
        result: Result[TypeReturn]
            Returns Ok containing the function result or otherwise Err containing the Excpetion.
        """
        try:
            return Ok(function())
        except Exception as e:
            return Err(e)

    @abc.abstractmethod
    def __str__(self: Result[TypeOk, TypeErr]) -> str:
        pass

    def __eq__(self: Result[TypeOk, TypeErr], other: object) -> bool:
        if isinstance(other, Err):
            return str(self) == str(other) and type(self._err_value) == type(other._err_value)  # type: ignore
        elif isinstance(other, Ok):
            return str(self) == str(other) and type(self._ok_value) == type(other._ok_value)  # type: ignore
        else:
            return False

    def __repr__(self: Result[TypeOk, TypeErr]) -> str:
        return str(self)


class Ok(Result[TypeOk, Any]):

    def __init__(self, value: TypeOk):
        self._ok_value = value
        self._err_value = Any
        self._is_err = False

    def __str__(self: Ok[TypeOk]) -> str:
        return f'Ok({self._ok_value})'


class Err(Result[Any, TypeErr]):

    def __init__(self, value: TypeErr):
        self._ok_value = Any
        self._err_value = value
        self._is_err = True

    def __str__(self: Err[TypeErr]) -> str:
        return f'Err({self._err_value})'


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
        if self._is_failure:
            return self._failure_value
        return default_value

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
