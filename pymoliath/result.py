"""
# Result Monad

The Result Monad is a container used to represent computations that can result in one of two
values: a success value or an error value. It encapsulates values that could be `Ok` (the
successful result) or `Err` (typically representing an error), allowing for a functional approach
to error handling by chaining operations without constant explicit error checks.

* Rust: [Result](https://doc.rust-lang.org/std/result/)
* Haskell: [Either](https://hackage.haskell.org/package/base-4.16.0.0/docs/Data-Either.html)

This implementation is heavily inspired by the Rust `Result` type - see `pymoliath.either` for the
sibling implementation using Haskell's `Left`/`Right` naming.

The `Result` type is a sum type that can be either `Ok` or `Err`.
In this implementation, it is represented as a Union type in Python.

```python
Result = Ok[TypeOk] | Err[TypeErr]
```

## Practical Examples and Benefits:

The Result Monad is particularly useful in scenarios where a function might fail and return an
error (e.g., looking up a key in a dictionary that doesn't exist, fetching a record from a database
that has been deleted, or parsing a string that doesn't match a specific format).

### Benefits:
1. Declarative Code: It allows you to chain operations together (using `map` and `bind`) without
   checking `if error` at every single step.
2. Error Propagation: If any step in a chain of operations returns `Err`, the subsequent operations
   are skipped automatically, and the final result will be `Err`.
3. Type Safety: It forces the developer to acknowledge the possibility of failure explicitly,
   making the code more robust against unhandled exceptions and making the flow of data more
   transparent.

#### Example: Fetching a user's profile and then their specific permission.

This approach turns "nested if"/try-except logic into a linear pipeline of transformations.

```python
# Without Result (Imperative)
try:
    user = get_user(user_id)
    profile = get_profile(user)
    permission = get_permission(profile)
    print(permission)
except UserNotFoundError:
    print("User not found")
except ProfileNotFoundError:
    print("Profile not found")
except PermissionDeniedError:
    print("Access denied")

# With Result (Functional)
(get_user(user_id)
    .map(get_profile)
    .map(get_permission)
    .unwrap_or_else(lambda error: f"Error: {error}"))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Result`
monad. Because the `Ok` and `Err` classes are designed to be compatible with Python's `match`
statement, you can easily branch your logic based on whether the operation succeeded or failed
without manually checking `is_ok()`/`is_err()`.

```python
match result_value:
    case Ok(x):
        # This block executes if the operation was successful
        print(f"Success value: {x}")
    case Err(y):
        # This block executes if the operation failed
        print(f"Error encountered: {y}")
```
"""

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
    """The Ok variant of the Result Monad.

    Result is a Monad that represents either success (Ok) or failure (Err). `Ok` represents the
    successful path of the computation and wraps the resulting value.
    """

    __slots__ = ("_ok_value",)
    __match_args__ = ("_ok_value",)

    def __init__(self, value: TypeOk):
        """Ok Monad constructor which takes a value of type TypeOk.

        Parameters
        ----------
        value: TypeOk
            Value to be stored in the Ok Monad.

        Examples
        --------
        >>> ok = Ok(42)
        >>> print(ok)
        Ok(42)
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.map(lambda x: x + 1)
        Ok(6)
        >>> names: Result[str, str] = Ok("alice")
        >>> names.map(str.upper)
        Ok(ALICE)
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.map_err(lambda e: e.upper())
        Ok(5)
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> def get_next(x: int) -> Result[int, str]:
        ...     return Ok(x + 1) if x < 10 else Err("too big")
        >>> val.bind(get_next)
        Ok(6)
        >>> Ok(10).bind(get_next)
        Err(too big)
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.bind_err(lambda e: Ok(0))
        Ok(5)
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> func: Result[Callable[[int], int], str] = Ok(lambda x: x * 2)
        >>> val.apply(func)
        Ok(20)
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

        Examples
        --------
        >>> func: Result[Callable[[int], int], str] = Ok(lambda y: 10 + y)
        >>> val: Result[int, str] = Ok(5)
        >>> func.apply2(val)
        Ok(15)
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.is_ok_and(lambda x: x > 5)
        True
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.is_err_and(lambda e: True)
        False
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
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

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.and_(val2)
        Ok(2)
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

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.or_(val2)
        Ok(1)
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

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.zip(val2)
        Ok((1, 2))
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

        Examples
        --------
        >>> val: Result[Result[int, str], str] = Ok(Ok(1))
        >>> val.flatten()
        Ok(1)
        """
        return cast(Result[Any, Any], self._ok_value)

    def ok(self) -> Option[TypeOk]:
        """Converts the Result Monad into an Option Monad, discarding any Err value.

        Returns
        -------
        option: Option[TypeOk]
            Returns Some with the Ok value.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.ok()
        Some(1)
        """
        from pymoliath.option import Some

        return Some(self._ok_value)

    def err(self) -> Option[Any]:
        """Converts the Result Monad into an Option Monad of the Err value, discarding the Ok value.

        Returns
        -------
        option: Option[TypeErr]
            Returns Nil, since this Result Monad is Ok.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.err()
        Nil()
        """
        from pymoliath.option import Nil

        return Nil()

    def unwrap(self) -> TypeOk:
        """Returns the Ok value if not Err, or otherwise raises an Exception with the Err value.

        Returns
        -------
        result: TypeOk
            Returns the Ok value or a default value.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap()
        1
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_or(0)
        1
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_or_else(lambda e: 0)
        1
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_err_or("default")
        'default'
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Ok(42)
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(42)
        >>> val.inspect_err(lambda e: print(f"Error: {e}"))
        Ok(42)
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

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.match(lambda e: "Error", lambda x: f"Ok: {x}")
        'Ok: 10'
        """
        return ok_function(self._ok_value)

    def is_ok(self) -> bool:
        """Returns True if the Result Monad is Ok, otherwise False if Err.

        Returns
        -------
        result: bool
            True: if Result Monad is Ok, False: if Result Monad is Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.is_ok()
        True
        """
        return True

    def is_err(self) -> bool:
        """Result monad is Err function

        Returns
        -------
        result: bool
            True: if Result Monad is Err, False: if Result Monad is Ok.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.is_err()
        False
        """
        return False

    def __str__(self) -> str:
        """Returns the string representation of the Ok Monad.

        Examples
        --------
        >>> str(Ok(42))
        'Ok(42)'
        """
        return f"Ok({self._ok_value})"

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is an equal Ok Monad (same wrapped value string and type).

        Examples
        --------
        >>> Ok(1) == Ok(1)
        True
        >>> Ok(1) == Ok(2)
        False
        """
        if isinstance(other, Ok):
            other_ok = cast(Ok[Any], other)
            return str(self) == str(other_ok) and type(self._ok_value) is type(
                other_ok._ok_value
            )
        return False

    def __repr__(self) -> str:
        """Returns the string representation of the Ok Monad (same as __str__).

        Examples
        --------
        >>> repr(Ok(10))
        'Ok(10)'
        """
        return str(self)


class Err(Generic[TypeErr]):
    """The Err variant of the Result Monad.

    Result is a Monad that represents either success (Ok) or failure (Err). `Err` represents the
    failed path of the computation and wraps the error value.
    """

    __slots__ = ("_err_value",)
    __match_args__ = ("_err_value",)

    def __init__(self, value: TypeErr):
        """Err Monad constructor which takes a value of type TypeErr.

        Parameters
        ----------
        value: TypeErr
            Value to be stored in the Err Monad.

        Examples
        --------
        >>> err = Err("Error Message")
        >>> print(err)
        Err(Error Message)
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.map(lambda x: x + 1)
        Err(Error)
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

        Examples
        --------
        >>> val: Result[int, str] = Err("error")
        >>> val.map_err(str.upper)
        Err(ERROR)
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.bind(lambda x: Ok(x + 1))
        Err(Error)
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.bind_err(lambda e: Ok(0))
        Ok(0)
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> func: Result[Callable[[int], int], str] = Ok(lambda x: x * 2)
        >>> val.apply(func)
        Err(Error)
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

        Examples
        --------
        >>> val: Result[Callable[[int], int], str] = Err("Error")
        >>> other: Result[int, str] = Ok(5)
        >>> val.apply2(other)
        Err(Error)
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.is_ok_and(lambda x: x > 5)
        False
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.is_err_and(lambda e: e == "Error")
        True
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.map_or("Fallback", lambda x: x)
        'Fallback'
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

        Examples
        --------
        >>> val1: Result[int, str] = Err("Error")
        >>> val2: Result[int, str] = Err("Other Error")
        >>> val1.and_(val2)
        Err(Error)
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

        Examples
        --------
        >>> val1: Result[int, str] = Err("Error")
        >>> val2: Result[int, str] = Ok(10)
        >>> val1.or_(val2)
        Ok(10)
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

        Examples
        --------
        >>> val1: Result[int, str] = Err("Error")
        >>> val2: Result[int, str] = Ok(10)
        >>> val1.zip(val2)
        Err(Error)
        """
        return self

    def flatten(self) -> Result[Any, TypeErr]:
        """Returns this Err unchanged, since there is nothing to flatten.

        Returns
        -------
        result: Result[TypeReturn, TypeErr]
            Returns this Err.

        Examples
        --------
        >>> val: Result[Result[int, str], str] = Err("Error")
        >>> val.flatten()
        Err(Error)
        """
        return self

    def ok(self) -> Option[Any]:
        """Converts the Result Monad into an Option Monad, discarding any Err value.

        Returns
        -------
        option: Option[TypeOk]
            Returns Nil, since this Result Monad is Err.

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.ok()
        Nil()
        """
        from pymoliath.option import Nil

        return Nil()

    def err(self) -> Option[TypeErr]:
        """Converts the Result Monad into an Option Monad of the Err value, discarding the Ok value.

        Returns
        -------
        option: Option[TypeErr]
            Returns Some with the Err value.

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.err()
        Some(Error)
        """
        from pymoliath.option import Some

        return Some(self._err_value)

    def unwrap(self) -> Any:
        """Raises an Exception containing the Err value, since this Result Monad is Err.

        Returns
        -------
        result: Any
            Never returns; always raises an Exception.

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.unwrap()
        Traceback (most recent call last):
            ...
        Exception: Error
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.unwrap_or(10)
        10
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.unwrap_or_else(lambda e: 0)
        0
        """
        return err_function(self._err_value)

    def unwrap_err_or(self, default_value: TypeErr) -> TypeErr:
        """Returns the Err value, since this Result Monad is Err.

        Parameters
        ----------
        default_value: TypeErr
            Default value of type TypeErr which is ignored since this Result Monad is Err.

        Returns
        -------
        result: TypeErr
            Returns the Err value.

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.unwrap_err_or("default")
        'Error'
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.inspect(lambda x: print(f"Value: {x}"))
        Err(Error)
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.inspect_err(lambda e: print(f"Error: {e}"))
        Error: Error
        Err(Error)
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

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.match(lambda e: f"Error: {e}", lambda x: "Ok")
        'Error: Error'
        """
        return err_function(self._err_value)

    def is_ok(self) -> bool:
        """Returns False, since this Result Monad is Err.

        Returns
        -------
        result: bool
            True: if Result Monad is Ok, False: if Result Monad is Err.

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.is_ok()
        False
        """
        return False

    def is_err(self) -> bool:
        """Returns True, since this Result Monad is Err.

        Returns
        -------
        result: bool
            True: if Result Monad is Err, False: if Result Monad is Ok.

        Examples
        --------
        >>> val: Result[int, str] = Err("Error")
        >>> val.is_err()
        True
        """
        return True

    def __str__(self) -> str:
        """Returns the string representation of the Err Monad.

        Examples
        --------
        >>> str(Err("Error"))
        'Err(Error)'
        """
        return f"Err({self._err_value})"

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is an equal Err Monad (same wrapped exception message and type).

        Examples
        --------
        >>> Err("Error") == Err("Error")
        True
        >>> Err("Error") == Err("Other")
        False
        """
        if isinstance(other, Err):
            other_err = cast(Err[Any], other)
            return str(self) == str(other_err) and type(self._err_value) is type(
                other_err._err_value
            )
        return False

    def __repr__(self) -> str:
        """Returns the string representation of the Err Monad (same as __str__).

        Examples
        --------
        >>> repr(Err("Error"))
        'Err(Error)'
        """
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

    Examples
    --------
    >>> def risky_call():
    ...     raise ValueError("Boom")
    >>> result_safe(risky_call)
    Err(Boom)
    >>> def safe_call():
    ...     return 42
    >>> result_safe(safe_call)
    Ok(42)
    """
    try:
        return Ok(function())
    except Exception as e:
        return Err(e)
