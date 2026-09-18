from __future__ import annotations

from functools import partial
from typing import Any, Callable, Generic, TypeAlias, TypeVar, cast

TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")

TypeReturn = TypeVar("TypeReturn")
TypeOk = TypeVar("TypeOk")
TypeErr = TypeVar("TypeErr")


class Ok(Generic[TypeOk]):
    """Result is a Monad that represents either success (Ok) or failure (Err)."""

    __match_args__ = ("_ok_value",)

    def __init__(self, value: TypeOk):
        self._ok_value = value

    def map(self, function: Callable[[TypeOk], TypeReturn]) -> Result[TypeReturn, Any]:
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
        return Ok(function(self._ok_value))

    def map_err(
        self, function: Callable[[Any], TypeReturn]
    ) -> Result[TypeOk, TypeReturn]:
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
        return self

    def bind(
        self, function: Callable[[TypeOk], Result[TypeReturn, Any]]
    ) -> Result[TypeReturn, Any]:
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
        return function(self._ok_value)

    def bind_err(
        self, function: Callable[[Any], Result[TypeOk, TypeReturn]]
    ) -> Result[TypeOk, TypeReturn]:
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
        return self

    def apply(
        self, applicative: Result[Callable[..., TypeReturn], Any]
    ) -> Result[TypeReturn, Any]:
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

        def binder(
            applicative_function: Callable[..., TypeReturn],
        ) -> Result[TypeReturn, Any]:
            def inner(x: Any) -> Any:
                try:
                    return applicative_function(x)
                except TypeError:
                    return partial(applicative_function, x)

            return self.map(inner)

        return applicative.bind(binder)

    def apply2(
        self: Ok[Callable[..., TypeReturn]],
        applicative_value: Result[Any, Any],
    ) -> Result[TypeReturn, Any]:
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

        def binder(
            applicative_function: Callable[..., TypeReturn],
        ) -> Result[TypeReturn, Any]:
            def inner(x: Any) -> Any:
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
        return self._ok_value

    def unwrap_or_else(self, err_function: Callable[[Any], TypeOk]) -> TypeOk:
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
        return default_value

    def inspect(self, function: Callable[[TypeOk], None]) -> Result[TypeOk, Any]:
        """Inspect the Result monad value of TypeOk

        Parameters
        ----------
        function: Callable[[TypeRight], None]
            Inspection function which takes the ok value of the Result monad

        Returns
        -------
        result: Result[TypeOk, TypeErr]
        """
        function(self._ok_value)
        return self

    def inspect_err(self, function: Callable[[Any], None]) -> Result[TypeOk, Any]:
        """Inspect the Result monad value of TypeErr

        Parameters
        ----------
        function: Callable[[TypeErr], None]
            Inspection function which takes the error value of the Result monad

        Returns
        -------
        result: Result[TypeOk, TypeErr]
        """
        return self

    def match(
        self,
        err_function: Callable[[Any], TypeReturn],
        ok_function: Callable[[TypeOk], TypeReturn],
    ) -> TypeReturn:
        """Matches the Result Monad to either an Err function or an Ok function with the same return type.

        Parameters
        ----------
        err_function: Callable[[Exception], TypeReturn]
            Callback function for either monads of type Err
        ok_function: Callable[[TypeOk], TypeReturn]
            Callback function for either monads of type Ok
        """
        return ok_function(self._ok_value)

    def is_ok(self) -> bool:
        """Returns True if the Result Monad is Ok, otherwise False if Err.

        Returns
        -------
        result: bool
            True: if Result Monad is Ok, False: if Result Monad is Err.
        """
        return True

    def is_err(self) -> bool:
        """Try monad is Err function

        Returns
        -------
        result: bool
            True: if Result Monad is Err, False: if Result Monad is Ok.
        """
        return False

    def __str__(self) -> str:
        return f"Ok({self._ok_value})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Ok):
            other_ok = cast(Ok[Any], other)
            return str(self) == str(other_ok) and type(self._ok_value) is type(
                other_ok._ok_value
            )
        return False

    def __repr__(self) -> str:
        return str(self)


class Err(Generic[TypeErr]):
    __match_args__ = ("_err_value",)

    def __init__(self, value: TypeErr):
        self._err_value = value

    def map(self, function: Callable[[Any], TypeReturn]) -> Result[TypeReturn, TypeErr]:
        return self

    def map_err(
        self, function: Callable[[TypeErr], TypeReturn]
    ) -> Result[Any, TypeReturn]:
        return Err(function(self._err_value))

    def bind(
        self, function: Callable[[Any], Result[TypeReturn, TypeErr]]
    ) -> Result[TypeReturn, TypeErr]:
        return self

    def bind_err(
        self, function: Callable[[TypeErr], Result[Any, TypeReturn]]
    ) -> Result[Any, TypeReturn]:
        return function(self._err_value)

    def apply(
        self, applicative: Result[Callable[..., TypeReturn], TypeErr]
    ) -> Result[TypeReturn, TypeErr]:
        return self

    def apply2(self, applicative_value: Result[Any, TypeErr]) -> Result[Any, TypeErr]:
        return self

    def unwrap(self) -> Any:
        raise Exception(self._err_value)

    def unwrap_or(self, default_value: TypeOk) -> TypeOk:
        return default_value

    def unwrap_or_else(self, err_function: Callable[[TypeErr], TypeOk]) -> TypeOk:
        return err_function(self._err_value)

    def unwrap_err_or(self, default_value: TypeErr) -> TypeErr:
        return self._err_value

    def inspect(self, function: Callable[[Any], None]) -> Result[Any, TypeErr]:
        return self

    def inspect_err(self, function: Callable[[TypeErr], None]) -> Result[Any, TypeErr]:
        function(self._err_value)
        return self

    def match(
        self,
        err_function: Callable[[TypeErr], TypeReturn],
        ok_function: Callable[[Any], TypeReturn],
    ) -> TypeReturn:
        return err_function(self._err_value)

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"Err({self._err_value})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Err):
            other_err = cast(Err[Any], other)
            return str(self) == str(other_err) and type(self._err_value) is type(
                other_err._err_value
            )
        return False

    def __repr__(self) -> str:
        return str(self)


Result: TypeAlias = Ok[TypeOk] | Err[TypeErr]


def result_safe(function: Callable[[], TypeReturn]) -> Result[TypeReturn, Exception]:
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
