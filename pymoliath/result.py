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

The `Result` type is a sealed sum type that can be either `Ok` or `Err`. Like in Rust, both
variants carry both type parameters:

```python
type Result[T, E] = Ok[T, E] | Err[T, E]
```

A bare `Ok(10)` is an `Ok[int, Never]`. A bare `Err("e")` leaves its Ok type open (`Err[Unknown, str]`)
so it can be solved from context, e.g. in `lambda x: Ok(x) if x > 0 else Err("negative")`.

## Typing like in Rust

Because both variants know the full `Result[T, E]`, lambdas passed to `map`/`bind`/... are
inferred cleanly and a `match` over `Ok`/`Err` is exhaustive. The one trade-off is invariance: a
bare `Ok(10)` without any context is an `Ok[int, Never]` and not assignable to a
`Result[int, str]`. Construct values directly in a `return` or an annotated assignment, just as
Rust needs a type annotation there:

```python
value: Result[int, str] = Ok(10)

def parse(text: str) -> Result[int, str]:
    return Ok(int(text)) if text.isdigit() else Err("not a number")
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
    .bind(get_profile)
    .bind(get_permission)
    .unwrap_or_else(lambda error: f"Error: {error}"))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Result`
monad. `Ok` and `Err` are the only variants (the Result is sealed), so a `match` over both is
exhaustive and type checkers narrow the value in each branch.

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

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Never,
    assert_never,
    cast,
    final,
    overload,
)

from typing_extensions import TypeVar

from pymoliath.util import curry

if TYPE_CHECKING:
    from pymoliath.option import Option

# Old-style TypeVars are invariant by default, which is what Result needs anyway
# (T and E also appear in parameter positions, e.g. unwrap_or and bind).
T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")

# E with a default of Never (PEP 696): a bare `Ok(10)` is an `Ok[int, Never]`, so - like in Rust - it
# needs an annotation before it can be combined with fallible code.
E_Never = TypeVar("E_Never", default=Never)


class _ResultImpl(Generic[T, E]):
    """Shared implementation of the Result Monad - `Ok` and `Err` are its only subclasses."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Runtime "sealed": only Ok and Err (defined in this module) may subclass.
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise TypeError("Result cannot be subclassed; use Ok or Err")

    def _as_result(self) -> Result[T, E]:
        # Safe: Ok and Err are the only subclasses (see __init_subclass__).
        return cast(Result[T, E], self)

    def map(self, function: Callable[[T], U]) -> Result[U, E]:
        """Calls function on a wrapped Ok value, otherwise leaving the Err value untouched.

        Parameters
        ----------
        function: Callable[[T], U]
            Function which takes a value of T and returns a value of type U.

        Returns
        -------
        result: Result[U, E]
            Returns an Ok with the function result or otherwise the Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.map(lambda x: x + 1)
        Ok(6)
        >>> err: Result[int, str] = Err("error")
        >>> err.map(lambda x: x + 1)
        Err(error)
        """
        match r := self._as_result():
            case Ok(value):
                return Ok(function(value))
            case Err(error):
                return Err(error)
            case _:
                assert_never(r)

    def map_err(self, function: Callable[[E], F]) -> Result[T, F]:
        """Calls function on a wrapped Err value, otherwise leaving the Ok value untouched.

        Parameters
        ----------
        function: Callable[[E], F]
            Function which takes a value of E and returns a value of F.

        Returns
        -------
        result: Result[T, F]
            Returns an Err with the function result or otherwise the Ok.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.map_err(lambda e: e.upper())
        Ok(5)
        >>> err: Result[int, str] = Err("error")
        >>> err.map_err(lambda e: e.upper())
        Err(ERROR)
        """
        match r := self._as_result():
            case Ok(value):
                return Ok(value)
            case Err(error):
                return Err(function(error))
            case _:
                assert_never(r)

    def bind(self, function: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Calls function if the Result Monad is Ok, otherwise returns the Err (Rust: `and_then`).

        Parameters
        ----------
        function: Callable[[T], Result[U, E]]
            Function which takes a value of T and returns a new Result Monad.

        Returns
        -------
        result: Result[U, E]
            Returns the function result if Ok, otherwise the Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> def get_next(x: int) -> Result[int, str]:
        ...     return Ok(x + 1) if x < 10 else Err("too big")
        >>> val.bind(get_next)
        Ok(6)
        >>> big: Result[int, str] = Ok(10)
        >>> big.bind(get_next)
        Err(too big)
        """
        match r := self._as_result():
            case Ok(value):
                return function(value)
            case Err(error):
                return Err(error)
            case _:
                assert_never(r)

    def bind_err(self, function: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Calls function if the Result Monad is Err, otherwise returns the Ok (Rust: `or_else`).

        Parameters
        ----------
        function: Callable[[E], Result[T, F]]
            Function which takes a value of E and returns a new Result Monad.

        Returns
        -------
        result: Result[T, F]
            Returns the function result if Err, otherwise the Ok.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.bind_err(lambda e: Ok(0))
        Ok(5)
        >>> err: Result[int, str] = Err("error")
        >>> err.bind_err(lambda e: Ok(len(e)))
        Ok(5)
        """
        match r := self._as_result():
            case Ok(value):
                return Ok(value)
            case Err(error):
                return function(error)
            case _:
                assert_never(r)

    def apply(self, applicative: Result[Callable[..., U], F]) -> Result[U, E | F]:
        """Applies the passed applicative wrapping a function if the Result Monad is Ok, otherwise
        returns the (first) Err. Functions of several arguments are curried.

        Unlike `bind`, the error types of both sides may differ (the result carries either), so
        bare applicatives such as `Ok(lambda x: x)` combine without annotations.

        Parameters
        ----------
        applicative: Result[Callable[[T], U], F]
            Applicative Result Monad which contains a function.

        Returns
        -------
        result: Result[U, E | F]
            Returns a Result Monad from the applied function if Ok, otherwise an Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> func: Result[Callable[[int], int], str] = Ok(lambda x: x * 2)
        >>> val.apply(func)
        Ok(20)
        """
        applicative_: Result[Callable[..., U], E | F] = cast(Any, applicative)
        self_: Result[T, E | F] = cast(Any, self)
        return applicative_.bind(lambda function: self_.map(curry(function)))

    def apply2(
        self: _ResultImpl[Callable[..., U], E], applicative_value: Result[Any, F]
    ) -> Result[U, E | F]:
        """Applies the function wrapped in this Result Monad to the passed Result Monad wrapping a
        value if both are Ok, otherwise returns the (first) Err. Functions of several arguments are
        curried. As with `apply`, the error types of both sides may differ.

        Parameters
        ----------
        applicative_value: Result[Any, F]
            Result monad which contains a value.

        Returns
        -------
        result: Result[U, E | F]
            Returns a Result Monad from the applied function if Ok, otherwise an Err.

        Examples
        --------
        >>> func: Result[Callable[[int], int], str] = Ok(lambda y: 10 + y)
        >>> val: Result[int, str] = Ok(5)
        >>> func.apply2(val)
        Ok(15)
        """
        self_: Result[Callable[..., U], E | F] = cast(Any, self)
        value_: Result[Any, E | F] = cast(Any, applicative_value)
        return self_.bind(lambda function: value_.map(curry(function)))

    def is_ok_and(self, function: Callable[[T], bool]) -> bool:
        """Returns True if the Result Monad is Ok and the predicate returns True for the value.

        Parameters
        ----------
        function: Callable[[T], bool]
            Predicate function applied to the Ok value.

        Returns
        -------
        result: bool
            Returns the predicate result if Ok, otherwise False.

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.is_ok_and(lambda x: x > 5)
        True
        """
        match r := self._as_result():
            case Ok(value):
                return function(value)
            case Err():
                return False
            case _:
                assert_never(r)

    def is_err_and(self, function: Callable[[E], bool]) -> bool:
        """Returns True if the Result Monad is Err and the predicate returns True for the error.

        Parameters
        ----------
        function: Callable[[E], bool]
            Predicate function applied to the Err value.

        Returns
        -------
        result: bool
            Returns the predicate result if Err, otherwise False.

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.is_err_and(lambda e: True)
        False
        """
        match r := self._as_result():
            case Ok():
                return False
            case Err(error):
                return function(error)
            case _:
                assert_never(r)

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        """Applies the function to the Ok value, or returns the default value if Err.

        Parameters
        ----------
        default_value: U
            Default value to be returned if the Result Monad is Err.
        function: Callable[[T], U]
            Function applied to the Ok value.

        Returns
        -------
        result: U
            Returns the function result or the default value.

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        """
        match r := self._as_result():
            case Ok(value):
                return function(value)
            case Err():
                return default_value
            case _:
                assert_never(r)

    def and_(self, other: Result[U, E]) -> Result[U, E]:
        """Returns `other` if the Result Monad is Ok, otherwise the Err.

        Parameters
        ----------
        other: Result[U, E]
            Result Monad to be returned if this Result Monad is Ok.

        Returns
        -------
        result: Result[U, E]
            Returns `other` or the Err.

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.and_(val2)
        Ok(2)
        """
        match r := self._as_result():
            case Ok():
                return other
            case Err(error):
                return Err(error)
            case _:
                assert_never(r)

    def or_(self, other: Result[T, F]) -> Result[T, F]:
        """Returns this Result Monad if it is Ok, otherwise `other`.

        Parameters
        ----------
        other: Result[T, F]
            Result Monad to be returned if this Result Monad is Err.

        Returns
        -------
        result: Result[T, F]
            Returns the Ok or `other`.

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.or_(val2)
        Ok(1)
        """
        match r := self._as_result():
            case Ok(value):
                return Ok(value)
            case Err():
                return other
            case _:
                assert_never(r)

    def zip(self, other: Result[U, E]) -> Result[tuple[T, U], E]:
        """Combines this Result Monad with another into a Result Monad of a tuple, or the first Err.

        Parameters
        ----------
        other: Result[U, E]
            Result Monad to be zipped with this Result Monad.

        Returns
        -------
        result: Result[tuple[T, U], E]
            Returns Ok of a tuple of both values, or Err.

        Examples
        --------
        >>> val1: Result[int, str] = Ok(1)
        >>> val2: Result[int, str] = Ok(2)
        >>> val1.zip(val2)
        Ok((1, 2))
        """
        return self.bind(
            lambda value: other.map(lambda other_value: (value, other_value))
        )

    @overload
    def flatten(self: _ResultImpl[Ok[U, F], E]) -> Result[U, E | F]: ...

    @overload
    def flatten(self: _ResultImpl[Err[U, F], E]) -> Result[U, E | F]: ...

    @overload
    def flatten(self: _ResultImpl[Result[U, F], E]) -> Result[U, E | F]: ...

    @overload
    def flatten(self: Err[Any, E]) -> Result[Never, E]: ...

    def flatten(self: _ResultImpl[Any, Any]) -> Result[Any, Any]:
        """Flattens a nested Result Monad by one level.

        Returns
        -------
        result: Result[U, E]
            Returns the nested Result Monad if Ok, otherwise the Err.

        Examples
        --------
        >>> Ok(Ok(1)).flatten()
        Ok(1)
        """
        match r := self._as_result():
            case Ok(value):
                return cast(Result[Any, Any], value)
            case Err(error):
                return Err(error)
            case _:
                assert_never(r)

    def ok(self) -> Option[T]:
        """Converts the Result Monad into an Option Monad, discarding any Err value.

        Returns
        -------
        option: Option[T]
            Returns Some with the Ok value, or Nil if Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.ok()
        Some(1)
        """
        from pymoliath.option import Nil, Some

        match r := self._as_result():
            case Ok(value):
                return Some(value)
            case Err():
                return Nil()
            case _:
                assert_never(r)

    def err(self) -> Option[E]:
        """Converts the Result Monad into an Option Monad of the Err value, discarding any Ok value.

        Returns
        -------
        option: Option[E]
            Returns Some with the Err value, or Nil if Ok.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.err()
        Nil()
        """
        from pymoliath.option import Nil, Some

        match r := self._as_result():
            case Ok():
                return Nil()
            case Err(error):
                return Some(error)
            case _:
                assert_never(r)

    def unwrap(self) -> T:
        """Returns the Ok value, or otherwise raises an Exception with the Err value.

        Returns
        -------
        result: T
            Returns the Ok value.

        Raises
        ------
        Exception
            If the Result Monad is Err.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap()
        1
        """
        match r := self._as_result():
            case Ok(value):
                return value
            case Err(error):
                raise Exception(error)
            case _:
                assert_never(r)

    def unwrap_or(self, default_value: T) -> T:
        """Returns the Ok value, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: T
            Default value of T

        Returns
        -------
        result: T
            Returns the Ok value or the default value.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_or(0)
        1
        """
        match r := self._as_result():
            case Ok(value):
                return value
            case Err():
                return default_value
            case _:
                assert_never(r)

    def unwrap_or_else(self, err_function: Callable[[E], T]) -> T:
        """Returns the Ok value, or otherwise calls the err_function with the Err value.

        Parameters
        ----------
        err_function: Callable[[E], T]
            Error function which will be called if the result is of type Err.

        Returns
        -------
        result: T
            Returns the Ok value or the err_function result.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_or_else(lambda e: 0)
        1
        """
        match r := self._as_result():
            case Ok(value):
                return value
            case Err(error):
                return err_function(error)
            case _:
                assert_never(r)

    def unwrap_err_or(self, default_value: E) -> E:
        """Returns the Err value, or otherwise a provided default value of E.

        Parameters
        ----------
        default_value: E
            Default value of E

        Returns
        -------
        result: E
            Returns the Err value or the default value.

        Examples
        --------
        >>> val: Result[int, str] = Ok(1)
        >>> val.unwrap_err_or("default")
        'default'
        """
        match r := self._as_result():
            case Ok():
                return default_value
            case Err(error):
                return error
            case _:
                assert_never(r)

    def inspect(self, function: Callable[[T], None]) -> Result[T, E]:
        """Calls function with the Ok value (if any) and returns the Result Monad unchanged.

        Parameters
        ----------
        function: Callable[[T], None]
            Inspection function which takes the Ok value of the Result monad

        Returns
        -------
        result: Result[T, E]

        Examples
        --------
        >>> val: Result[int, str] = Ok(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Ok(42)
        """
        match r := self._as_result():
            case Ok(value):
                function(value)
            case Err():
                pass
            case _:
                assert_never(r)
        return r

    def inspect_err(self, function: Callable[[E], None]) -> Result[T, E]:
        """Calls function with the Err value (if any) and returns the Result Monad unchanged.

        Parameters
        ----------
        function: Callable[[E], None]
            Inspection function which takes the Err value of the Result monad

        Returns
        -------
        result: Result[T, E]

        Examples
        --------
        >>> val: Result[int, str] = Err("boom")
        >>> val.inspect_err(lambda e: print(f"Error: {e}"))
        Error: boom
        Err(boom)
        """
        match r := self._as_result():
            case Ok():
                pass
            case Err(error):
                function(error)
            case _:
                assert_never(r)
        return r

    def match(self, err_function: Callable[[E], U], ok_function: Callable[[T], U]) -> U:
        """Matches the Result Monad to either an Err function or an Ok function with the same
        return type.

        Parameters
        ----------
        err_function: Callable[[E], U]
            Callback function for Result monads of type Err
        ok_function: Callable[[T], U]
            Callback function for Result monads of type Ok

        Examples
        --------
        >>> val: Result[int, str] = Ok(10)
        >>> val.match(lambda e: "Error", lambda x: f"Ok: {x}")
        'Ok: 10'
        """
        match r := self._as_result():
            case Ok(value):
                return ok_function(value)
            case Err(error):
                return err_function(error)
            case _:
                assert_never(r)

    def is_ok(self) -> bool:
        """Returns True if the Result Monad is Ok, otherwise False.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.is_ok()
        True
        """
        return isinstance(self, Ok)

    def is_err(self) -> bool:
        """Returns True if the Result Monad is Err, otherwise False.

        Examples
        --------
        >>> val: Result[int, str] = Ok(5)
        >>> val.is_err()
        False
        """
        return isinstance(self, Err)

    def _payload(self) -> object:
        match r := self._as_result():
            case Ok(value):
                return value
            case Err(error):
                return error
            case _:
                assert_never(r)

    def __str__(self) -> str:
        """Returns the string representation of the Result Monad.

        Examples
        --------
        >>> str(Ok(42))
        'Ok(42)'
        >>> str(Err("error"))
        'Err(error)'
        """
        return f"{type(self).__name__}({self._payload()})"

    def __repr__(self) -> str:
        """Returns the string representation of the Result Monad (same as __str__)."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is the same variant wrapping a value of the same type and string.

        Examples
        --------
        >>> Ok(1) == Ok(1)
        True
        >>> Ok(1) == Err(1)
        False
        """
        if type(self) is not type(other):
            return False
        value, other_value = (
            self._payload(),
            cast(_ResultImpl[Any, Any], other)._payload(),
        )
        return type(value) is type(other_value) and str(value) == str(other_value)

    __hash__ = None  # type: ignore[assignment]


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Ok(_ResultImpl[T, E_Never]):
    value: T


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Err(_ResultImpl[T, E]):
    error: E


# Err's Ok type has deliberately no default: if it defaulted to Never, pyright would pin a lambda's
# Ok type to Never as soon as one branch returns an Err (`lambda x: Ok(x) if x else Err("zero")`).
type Result[OkT, ErrT] = Ok[OkT, ErrT] | Err[OkT, ErrT]


def result_safe(function: Callable[[], T]) -> Result[T, Exception]:
    """Calls function and wraps its return value in Ok, or a raised Exception in Err.

    Parameters
    ----------
    function: Callable[[], T]
        Zero-argument function which may raise an Exception.

    Returns
    -------
    result: Result[T, Exception]

    Examples
    --------
    >>> result_safe(lambda: 1)
    Ok(1)
    >>> result_safe(lambda: 1 / 0)
    Err(division by zero)
    """
    try:
        return Ok(function())
    except Exception as e:
        return Err(e)
