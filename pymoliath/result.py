from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from pymoliath.util import curry

if TYPE_CHECKING:
    from pymoliath.option import Option

TypeResult = TypeVar("TypeResult")
TypePure = TypeVar("TypePure")

TypeReturn = TypeVar("TypeReturn")
TypeOk = TypeVar("TypeOk")
TypeErr = TypeVar("TypeErr")


class Ok(Generic[TypeOk]):
    """Result is a Monad that represents either success (Ok) or failure (Err)."""

    __slots__ = ("_ok_value",)
    __match_args__ = ("_ok_value",)

    def __init__(self, value: TypeOk):
        """Ok Monad constructor which takes a value of type TypeOk.

        Parameters
        ----------
        value: TypeOk
            Value to be stored in the Ok Monad.
        """
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
            """Maps the applicative's function, curried, over this Ok value."""
            return self.map(curry(applicative_function))

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
            """Maps the applicative's function, curried, over the applicative_value."""
            return applicative_value.map(curry(applicative_function))

        return self.bind(binder)

    def is_ok_and(self, function: Callable[[TypeOk], bool]) -> bool:
        """Returns True if the Result Monad is Ok and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[TypeOk], bool]
            Predicate function applied to the Ok value.

        Returns
        -------
        result: bool
            Returns the predicate result.
        """
        return function(self._ok_value)

    def is_err_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns False, since this Result Monad is Ok.

        Parameters
        ----------
        function: Callable[[TypeErr], bool]
            Predicate function which would be applied to the Err value.

        Returns
        -------
        result: bool
            Returns False.
        """
        return False

    def map_or(
        self, default_value: TypeReturn, function: Callable[[TypeOk], TypeReturn]
    ) -> TypeReturn:
        """Applies the function to the Ok value, or returns the default value if Err.

        Parameters
        ----------
        default_value: TypeReturn
            Default value to be returned if the Result Monad is Err.
        function: Callable[[TypeOk], TypeReturn]
            Function applied to the Ok value.

        Returns
        -------
        result: TypeReturn
            Returns the function result.
        """
        return function(self._ok_value)

    def and_(self, other: Result[TypeReturn, Any]) -> Result[TypeReturn, Any]:
        """Returns `other` if the Result Monad is Ok, otherwise Err.

        Parameters
        ----------
        other: Result[TypeReturn, TypeErr]
            Result Monad to be returned if this Result Monad is Ok.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns `other`.
        """
        return other

    def or_(self, other: Result[TypeOk, Any]) -> Result[TypeOk, Any]:
        """Returns this Result Monad if it is Ok, otherwise `other`.

        Parameters
        ----------
        other: Result[TypeOk, TypeErr]
            Result Monad to be returned if this Result Monad is Err.

        Returns
        -------
        result: Result[TypeOk, TypeErr]
            Returns this Ok.
        """
        return self

    def zip(self, other: Result[TypePure, Any]) -> Result[Tuple[TypeOk, TypePure], Any]:
        """Combines this Result Monad with another into a Result Monad of a tuple, or Err if either is Err.

        Parameters
        ----------
        other: Result[TypePure, TypeErr]
            Result Monad to be zipped with this Result Monad.

        Returns
        -------
        result: Result[Tuple[TypeOk, TypePure], TypeErr]
            Returns Ok of a tuple of both values, or Err.
        """
        return other.map(lambda o: (self._ok_value, o))

    @overload
    def flatten(self: Ok[Ok[TypeReturn]]) -> Result[TypeReturn, Any]: ...

    @overload
    def flatten(self: Ok[Err[TypeErr]]) -> Result[Any, TypeErr]: ...

    def flatten(self) -> Result[Any, Any]:
        """Flattens a nested Result Monad by one level.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns the nested Result Monad.
        """
        return cast(Result[Any, Any], self._ok_value)

    def ok(self) -> Option[TypeOk]:
        """Converts the Result Monad into an Option Monad, discarding any Err value.

        Returns
        -------
        option: Option[TypeOk]
            Returns Some with the Ok value.
        """
        from pymoliath.option import Some

        return Some(self._ok_value)

    def err(self) -> Option[Any]:
        """Converts the Result Monad into an Option Monad of the Err value, discarding the Ok value.

        Returns
        -------
        option: Option[TypeErr]
            Returns Nil, since this Result Monad is Ok.
        """
        from pymoliath.option import Nil

        return Nil()

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
        """Returns the string representation of the Ok Monad."""
        return f"Ok({self._ok_value})"

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is an equal Ok Monad (same wrapped value string and type)."""
        if isinstance(other, Ok):
            other_ok = cast(Ok[Any], other)
            return str(self) == str(other_ok) and type(self._ok_value) is type(
                other_ok._ok_value
            )
        return False

    def __repr__(self) -> str:
        """Returns the string representation of the Ok Monad (same as __str__)."""
        return str(self)


class Err(Generic[TypeErr]):
    __slots__ = ("_err_value",)
    __match_args__ = ("_err_value",)

    def __init__(self, value: TypeErr):
        """Err Monad constructor which takes a value of type TypeErr.

        Parameters
        ----------
        value: TypeErr
            Value to be stored in the Err Monad.
        """
        self._err_value = value

    def map(self, function: Callable[[Any], TypeReturn]) -> Result[TypeReturn, TypeErr]:
        """Returns this Err unchanged, since map only transforms the Ok value.

        Parameters
        ----------
        function: Callable[[TypeOk], TypeReturn]
            Function which would be applied to the Ok value if this Result Monad were Ok.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns this Err.
        """
        return self

    def map_err(
        self, function: Callable[[TypeErr], TypeReturn]
    ) -> Result[Any, TypeReturn]:
        """Calls function on the wrapped Err value and returns a new Err with the result.

        Parameters
        ----------
        function: Callable[[TypeErr], TypeReturn]
            Function which takes a value of TypeErr and returns a value of type TypeReturn.

        Returns
        -------
        result: Result[TypeOk, TypeReturn]
            Returns a new Err with the function result.
        """
        return Err(function(self._err_value))

    def bind(
        self, function: Callable[[Any], Result[TypeReturn, TypeErr]]
    ) -> Result[TypeReturn, TypeErr]:
        """Returns this Err unchanged, since bind only chains on the Ok value.

        Parameters
        ----------
        function: Callable[[TypeOk], Result[TypeReturn, TypeErr]]
            Function which would be called with the Ok value if this Result Monad were Ok.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns this Err.
        """
        return self

    def bind_err(
        self, function: Callable[[TypeErr], Result[Any, TypeReturn]]
    ) -> Result[Any, TypeReturn]:
        """Calls function with the wrapped Err value and returns its resulting Result Monad.

        Parameters
        ----------
        function: Callable[[TypeErr], Result[TypeOk, TypeReturn]]
            Function which takes a value of TypeErr and returns a new Result Monad.

        Returns
        -------
        result: Result[TypeOk, TypeReturn]
            Returns the Result Monad from the function call.
        """
        return function(self._err_value)

    def apply(
        self, applicative: Result[Callable[..., TypeReturn], TypeErr]
    ) -> Result[TypeReturn, TypeErr]:
        """Returns this Err unchanged, since apply only executes the applicative's function for Ok.

        Parameters
        ----------
        applicative: Result[Callable[[TypeOk], TypeReturn], TypeErr]
            Applicative Result Monad which contains a function.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns this Err.
        """
        return self

    def apply2(self, applicative_value: Result[Any, TypeErr]) -> Result[Any, TypeErr]:
        """Returns this Err unchanged, since apply2 only applies the value when this Result Monad is Ok.

        Parameters
        ----------
        applicative_value: Result[TypePure, TypeErr]
            Result Monad which contains a value.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns this Err.
        """
        return self

    def is_ok_and(self, function: Callable[[Any], bool]) -> bool:
        """Returns False, since this Result Monad is Err.

        Parameters
        ----------
        function: Callable[[TypeOk], bool]
            Predicate function which would be applied to the Ok value.

        Returns
        -------
        result: bool
            Returns False.
        """
        return False

    def is_err_and(self, function: Callable[[TypeErr], bool]) -> bool:
        """Returns True if the Result Monad is Err and the predicate returns True for the contained value.

        Parameters
        ----------
        function: Callable[[TypeErr], bool]
            Predicate function applied to the Err value.

        Returns
        -------
        result: bool
            Returns the predicate result.
        """
        return function(self._err_value)

    def map_or(
        self, default_value: TypeReturn, function: Callable[[Any], TypeReturn]
    ) -> TypeReturn:
        """Returns the default value, since this Result Monad is Err.

        Parameters
        ----------
        default_value: TypeReturn
            Default value to be returned since the Result Monad is Err.
        function: Callable[[TypeOk], TypeReturn]
            Function which would be applied to the Ok value if this Result Monad were Ok.

        Returns
        -------
        result: TypeReturn
            Returns the default value.
        """
        return default_value

    def and_(self, other: Result[Any, TypeErr]) -> Result[Any, TypeErr]:
        """Returns this Err, since and_ only returns `other` when this Result Monad is Ok.

        Parameters
        ----------
        other: Result[TypeReturn, TypeErr]
            Result Monad which would be returned if this Result Monad were Ok.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns this Err.
        """
        return self

    def or_(self, other: Result[TypeOk, Any]) -> Result[TypeOk, Any]:
        """Returns `other`, since this Result Monad is Err.

        Parameters
        ----------
        other: Result[TypeOk, TypeErr]
            Result Monad to be returned since this Result Monad is Err.

        Returns
        -------
        result: Result[TypeOk, TypeErr]
            Returns `other`.
        """
        return other

    def zip(self, other: Result[Any, Any]) -> Result[Any, TypeErr]:
        """Returns this Err, since zip cannot combine values when this Result Monad is Err.

        Parameters
        ----------
        other: Result[TypePure, TypeErr]
            Result Monad which would be zipped with this Result Monad if it were Ok.

        Returns
        -------
        result: Result[Tuple[TypeOk, TypePure], TypeErr]
            Returns this Err.
        """
        return self

    def flatten(self) -> Result[Any, TypeErr]:
        """Returns this Err unchanged, since there is nothing to flatten.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns this Err.
        """
        return self

    def ok(self) -> Option[Any]:
        """Converts the Result Monad into an Option Monad, discarding any Err value.

        Returns
        -------
        option: Option[TypeOk]
            Returns Nil, since this Result Monad is Err.
        """
        from pymoliath.option import Nil

        return Nil()

    def err(self) -> Option[TypeErr]:
        """Converts the Result Monad into an Option Monad of the Err value, discarding the Ok value.

        Returns
        -------
        option: Option[TypeErr]
            Returns Some with the Err value.
        """
        from pymoliath.option import Some

        return Some(self._err_value)

    def unwrap(self) -> Any:
        """Raises an Exception containing the Err value, since this Result Monad is Err.

        Returns
        -------
        result: Any
            Never returns; always raises an Exception.
        """
        raise Exception(self._err_value)

    def unwrap_or(self, default_value: TypeOk) -> TypeOk:
        """Returns the provided default value, since this Result Monad is Err.

        Parameters
        ----------
        default_value: TypeOk
            Default value of TypeOk

        Returns
        -------
        result: TypeOk
            Returns the default value.
        """
        return default_value

    def unwrap_or_else(self, err_function: Callable[[TypeErr], TypeOk]) -> TypeOk:
        """Calls err_function with the Err value and returns its result, since this Result Monad is Err.

        Parameters
        ----------
        err_function: Callable[[TypeErr], TypeOk]
            Error function which will be called with the Err value.

        Returns
        -------
        result: TypeOk
            Returns the result of calling err_function with the Err value.
        """
        return err_function(self._err_value)

    def unwrap_err_or(self, default_value: TypeErr) -> TypeErr:
        """Returns the Err value, since this Result Monad is Err.

        Parameters
        ----------
        default_value: TypeErr
            Default value of TypeErr which is ignored since this Result Monad is Err.

        Returns
        -------
        result: TypeErr
            Returns the Err value.
        """
        return self._err_value

    def inspect(self, function: Callable[[Any], None]) -> Result[Any, TypeErr]:
        """Returns this Err unchanged, without calling function, since this Result Monad is Err.

        Parameters
        ----------
        function: Callable[[TypeOk], None]
            Inspection function which would be called with the Ok value if this Result Monad were Ok.

        Returns
        -------
        result: Result[TypeOk, TypeErr]
        """
        return self

    def inspect_err(self, function: Callable[[TypeErr], None]) -> Result[Any, TypeErr]:
        """Inspect the Result monad value of TypeErr

        Parameters
        ----------
        function: Callable[[TypeErr], None]
            Inspection function which takes the error value of the Result monad

        Returns
        -------
        result: Result[TypeOk, TypeErr]
        """
        function(self._err_value)
        return self

    def match(
        self,
        err_function: Callable[[TypeErr], TypeReturn],
        ok_function: Callable[[Any], TypeReturn],
    ) -> TypeReturn:
        """Matches the Result Monad to either an Err function or an Ok function with the same return type.

        Parameters
        ----------
        err_function: Callable[[Exception], TypeReturn]
            Callback function for either monads of type Err
        ok_function: Callable[[TypeOk], TypeReturn]
            Callback function for either monads of type Ok
        """
        return err_function(self._err_value)

    def is_ok(self) -> bool:
        """Returns False, since this Result Monad is Err.

        Returns
        -------
        result: bool
            True: if Result Monad is Ok, False: if Result Monad is Err.
        """
        return False

    def is_err(self) -> bool:
        """Returns True, since this Result Monad is Err.

        Returns
        -------
        result: bool
            True: if Result Monad is Err, False: if Result Monad is Ok.
        """
        return True

    def __str__(self) -> str:
        """Returns the string representation of the Err Monad."""
        return f"Err({self._err_value})"

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is an equal Err Monad (same wrapped exception message and type)."""
        if isinstance(other, Err):
            other_err = cast(Err[Any], other)
            return str(self) == str(other_err) and type(self._err_value) is type(
                other_err._err_value
            )
        return False

    def __repr__(self) -> str:
        """Returns the string representation of the Err Monad (same as __str__)."""
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
