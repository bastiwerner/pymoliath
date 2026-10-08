"""
# Try Monad

The Try Monad is a container used to represent computations that either succeed with a value or
fail by raising an exception. It encapsulates the outcome of code that might raise, allowing for a
functional approach to exception handling by chaining operations without wrapping every step in
its own `try`/`except` block. Unlike `Result`, whose `Ok`/`Err` values are constructed explicitly
by the caller, `Try`'s `map`/`bind` automatically catch any `Exception` raised by the function they
are given and turn it into a `Failure`.

* Scala: [Try](https://www.scala-lang.org/api/current/scala/util/Try.html)

This implementation is heavily inspired by Scala's `Try` type.

The `Try` type is a sealed sum type that can be either `Success` or `Failure`. Like Rust's
`Result`, both variants carry the type parameter:

```python
type Try[T] = Success[T] | Failure[T]
```

A bare `Success(10)` is a `Success[int]`. A bare `Failure(e)` leaves its success type open
(`Failure[Unknown]`) so it can be solved from context, e.g. in
`lambda x: Success(x) if x > 0 else Failure(ValueError("negative"))`.

## Practical Examples and Benefits:

The Try Monad is particularly useful in scenarios where a chain of operations might each raise an
exception (e.g. parsing a string, calling out to a library that isn't exception-free, or performing
several arithmetic operations that could divide by zero), and you want the first exception to short
-circuit the rest of the chain without a `try`/`except` around each step.

### Benefits:
1. Declarative Code: `map`/`bind` chain operations together while automatically catching any
   exception the passed function raises, rather than requiring a `try`/`except` per step.
2. Error Propagation: If any step in a chain of operations produces a `Failure`, the subsequent
   operations are skipped automatically, and the final result stays a `Failure`.
3. Type Safety: It forces the developer to acknowledge the possibility of failure explicitly in the
   return type, rather than having exceptions propagate implicitly and invisibly through the call
   stack.

#### Example: Parsing and transforming a value that might raise.

```python
# Without Try (Imperative)
try:
    parsed = int(raw_value)
    doubled = parsed * 2
    print(doubled)
except ValueError as e:
    print(f"Error: {e}")

# With Try (Functional)
(safe(lambda: int(raw_value))
    .map(lambda parsed: parsed * 2)
    .match(lambda e: print(f"Error: {e}"), print))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Try`
monad. `Success` and `Failure` are the only variants (the Try is sealed), so a `match` over both is
exhaustive and type checkers narrow the value in each branch, without manually checking
`is_success()`/`is_failure()`.

```python
match try_value:
    case Success(x):
        # This block executes if the computation succeeded
        print(f"Success value: {x}")
    case Failure(e):
        # This block executes if the computation raised
        print(f"Exception encountered: {e}")
```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Never, assert_never, cast, final, overload

from typing_extensions import TypeVar

from pymoliath.either import Either, Left, Right
from pymoliath.result import Err, Ok, Result
from pymoliath.util import curry

# Old-style TypeVars are invariant by default, which is what Try needs anyway
# (T also appears in parameter positions, e.g. unwrap_or and or_).
T = TypeVar("T")
U = TypeVar("U")


class _TryImpl(Generic[T]):
    """Shared implementation of the Try Monad - `Success` and `Failure` are its only subclasses."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Runtime "sealed": only Success and Failure (defined in this module) may subclass.
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise TypeError("Try cannot be subclassed; use Success or Failure")

    def _as_try(self) -> Try[T]:
        # Safe: Success and Failure are the only subclasses (see __init_subclass__).
        return cast("Try[T]", self)

    def map(self, function: Callable[[T], U]) -> Try[U]:
        """Calls function on a wrapped Success value, otherwise leaving the Failure untouched.
        Any exception raised by the function is caught and turned into a Failure.

        Parameters
        ----------
        function: Callable[[T], U]
            Function which takes a value of T and returns a value of type U.

        Returns
        -------
        try: Try[U]
            Returns a Success with the function result or otherwise a Failure.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.map(lambda x: x + 1)
        Success(6)
        >>> def risky(x: int) -> int:
        ...     raise ValueError("boom")
        >>> val.map(risky)
        Failure(boom)
        """
        match t := self._as_try():
            case Success(value):
                try:
                    return Success(function(value))
                except Exception as e:
                    return Failure(e)
            case Failure(exception):
                return Failure(exception)
            case _:
                assert_never(t)

    def map_failure(self, function: Callable[[Exception], Exception]) -> Try[T]:
        """Calls function on a wrapped Failure exception, otherwise leaving the Success untouched.
        Any exception raised by the function is caught and turned into a Failure.

        Parameters
        ----------
        function: Callable[[Exception], Exception]
            Function which takes an Exception and returns an Exception.

        Returns
        -------
        try: Try[T]
            Returns a Failure with the function result or otherwise the Success.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.map_failure(lambda e: ValueError("wrapped"))
        Success(5)
        >>> failed: Try[int] = Failure(ValueError("boom"))
        >>> failed.map_failure(lambda e: ValueError(f"wrapped {e}"))
        Failure(wrapped boom)
        """
        match t := self._as_try():
            case Success(value):
                return Success(value)
            case Failure(exception):
                try:
                    return Failure(function(exception))
                except Exception as e:
                    return Failure(e)
            case _:
                assert_never(t)

    def bind(self, function: Callable[[T], Try[U]]) -> Try[U]:
        """Calls function if the Try Monad is Success, otherwise returns the Failure.
        Any exception raised by the function is caught and turned into a Failure.

        Parameters
        ----------
        function: Callable[[T], Try[U]]
            Function which takes a value of T and returns a new Try Monad.

        Returns
        -------
        try: Try[U]
            Returns the function result if Success, otherwise a Failure.

        Examples
        --------
        >>> def get_next(x: int) -> Try[int]:
        ...     return Success(x + 1) if x < 10 else Failure(ValueError("too big"))
        >>> Success(5).bind(get_next)
        Success(6)
        >>> Success(10).bind(get_next)
        Failure(too big)
        """
        match t := self._as_try():
            case Success(value):
                try:
                    return function(value)
                except Exception as e:
                    return Failure(e)
            case Failure(exception):
                return Failure(exception)
            case _:
                assert_never(t)

    def bind_failure(self, function: Callable[[Exception], Try[T]]) -> Try[T]:
        """Calls function if the Try Monad is Failure, otherwise returns the Success.
        Any exception raised by the function is caught and turned into a Failure.

        Parameters
        ----------
        function: Callable[[Exception], Try[T]]
            Function which takes the Exception and returns a new Try Monad.

        Returns
        -------
        try: Try[T]
            Returns the function result if Failure, otherwise the Success.

        Examples
        --------
        >>> failed: Try[int] = Failure(ValueError("boom"))
        >>> failed.bind_failure(lambda e: Success(0))
        Success(0)
        """
        match t := self._as_try():
            case Success(value):
                return Success(value)
            case Failure(exception):
                try:
                    return function(exception)
                except Exception as e:
                    return Failure(e)
            case _:
                assert_never(t)

    def apply(self, applicative: Try[Callable[..., U]]) -> Try[U]:
        """Applies the passed applicative wrapping a function if the Try Monad is Success,
        otherwise returns the (first) Failure. Functions of several arguments are curried.

        Parameters
        ----------
        applicative: Try[Callable[[T], U]]
            Applicative Try Monad which contains a function.

        Returns
        -------
        try: Try[U]
            Returns a Try Monad from the applied function if Success, otherwise a Failure.

        Examples
        --------
        >>> func: Try[Callable[[int], int]] = Success(lambda x: x * 2)
        >>> val: Try[int] = Success(10)
        >>> val.apply(func)
        Success(20)
        """
        return applicative.bind(lambda function: self.map(curry(function)))

    def apply2(self: _TryImpl[Callable[..., U]], applicative_value: Try[Any]) -> Try[U]:
        """Applies the function wrapped in this Try Monad to the passed Try Monad wrapping a value
        if both are Success, otherwise returns the (first) Failure. Functions of several arguments
        are curried.

        Parameters
        ----------
        applicative_value: Try[Any]
            Try Monad which contains a value.

        Returns
        -------
        try: Try[U]
            Returns a Try Monad from the applied function if Success, otherwise a Failure.

        Examples
        --------
        >>> func: Try[Callable[[int], int]] = Success(lambda x: x * 2)
        >>> val: Try[int] = Success(10)
        >>> func.apply2(val)
        Success(20)
        """
        return self.bind(lambda function: applicative_value.map(curry(function)))

    def is_success_and(self, function: Callable[[T], bool]) -> bool:
        """Returns True if the Try Monad is Success and the predicate returns True for the value.

        Parameters
        ----------
        function: Callable[[T], bool]
            Predicate function applied to the Success value.

        Returns
        -------
        result: bool
            Returns the predicate result if Success, otherwise False.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.is_success_and(lambda x: x > 5)
        True
        """
        match t := self._as_try():
            case Success(value):
                return function(value)
            case Failure():
                return False
            case _:
                assert_never(t)

    def is_failure_and(self, function: Callable[[Exception], bool]) -> bool:
        """Returns True if the Try Monad is Failure and the predicate returns True for the
        exception.

        Parameters
        ----------
        function: Callable[[Exception], bool]
            Predicate function applied to the Failure exception.

        Returns
        -------
        result: bool
            Returns the predicate result if Failure, otherwise False.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.is_failure_and(lambda e: isinstance(e, ValueError))
        True
        """
        match t := self._as_try():
            case Success():
                return False
            case Failure(exception):
                return function(exception)
            case _:
                assert_never(t)

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        """Applies the function to the Success value, or returns the default value if Failure.

        Parameters
        ----------
        default_value: U
            Default value to be returned if the Try Monad is Failure.
        function: Callable[[T], U]
            Function applied to the Success value.

        Returns
        -------
        result: U
            Returns the function result or the default value.

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.map_or(0, lambda x: x * 2)
        20
        """
        match t := self._as_try():
            case Success(value):
                return function(value)
            case Failure():
                return default_value
            case _:
                assert_never(t)

    def and_(self, other: Try[U]) -> Try[U]:
        """Returns `other` if the Try Monad is Success, otherwise the Failure.

        Parameters
        ----------
        other: Try[U]
            Try Monad to be returned if this Try Monad is Success.

        Returns
        -------
        try: Try[U]
            Returns `other` or the Failure.

        Examples
        --------
        >>> val1: Try[int] = Success(1)
        >>> val2: Try[int] = Success(2)
        >>> val1.and_(val2)
        Success(2)
        """
        match t := self._as_try():
            case Success():
                return other
            case Failure(exception):
                return Failure(exception)
            case _:
                assert_never(t)

    def or_(self, other: Try[T]) -> Try[T]:
        """Returns this Try Monad if it is Success, otherwise `other`.

        Parameters
        ----------
        other: Try[T]
            Try Monad to be returned if this Try Monad is Failure.

        Returns
        -------
        try: Try[T]
            Returns the Success or `other`.

        Examples
        --------
        >>> val1: Try[int] = Failure(ValueError("boom"))
        >>> val2: Try[int] = Success(2)
        >>> val1.or_(val2)
        Success(2)
        """
        match t := self._as_try():
            case Success(value):
                return Success(value)
            case Failure():
                return other
            case _:
                assert_never(t)

    def zip(self, other: Try[U]) -> Try[tuple[T, U]]:
        """Combines this Try Monad with another into a Try Monad of a tuple, or the first Failure.

        Parameters
        ----------
        other: Try[U]
            Try Monad to be zipped with this Try Monad.

        Returns
        -------
        try: Try[tuple[T, U]]
            Returns Success of a tuple of both values, or Failure.

        Examples
        --------
        >>> val1: Try[int] = Success(1)
        >>> val2: Try[int] = Success(2)
        >>> val1.zip(val2)
        Success((1, 2))
        """
        return self.bind(
            lambda value: other.map(lambda other_value: (value, other_value))
        )

    @overload
    def flatten(self: _TryImpl[Success[U]]) -> Try[U]: ...

    @overload
    def flatten(self: _TryImpl[Failure[U]]) -> Try[U]: ...

    @overload
    def flatten(self: _TryImpl[Try[U]]) -> Try[U]: ...

    @overload
    def flatten(self: Failure[Any]) -> Try[Never]: ...

    def flatten(self: _TryImpl[Any]) -> Try[Any]:
        """Flattens a nested Try Monad by one level.

        Returns
        -------
        try: Try[U]
            Returns the nested Try Monad if Success, otherwise the Failure.

        Examples
        --------
        >>> Success(Success(1)).flatten()
        Success(1)
        """
        match t := self._as_try():
            case Success(value):
                return cast("Try[Any]", value)
            case Failure(exception):
                return Failure(exception)
            case _:
                assert_never(t)

    def unwrap(self) -> T:
        """Returns the Success value, or otherwise raises the wrapped exception.

        Returns
        -------
        result: T
            Returns the Success value.

        Raises
        ------
        Exception
            The wrapped exception if the Try Monad is Failure.

        Examples
        --------
        >>> val: Try[int] = Success(1)
        >>> val.unwrap()
        1
        """
        match t := self._as_try():
            case Success(value):
                return value
            case Failure(exception):
                raise exception
            case _:
                assert_never(t)

    def unwrap_or(self, default_value: T) -> T:
        """Returns the Success value, or otherwise a provided default value of the same type.

        Parameters
        ----------
        default_value: T
            Default value of T

        Returns
        -------
        result: T
            Returns the Success value or the default value.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.unwrap_or(0)
        0
        """
        match t := self._as_try():
            case Success(value):
                return value
            case Failure():
                return default_value
            case _:
                assert_never(t)

    def unwrap_or_else(self, failure_function: Callable[[Exception], T]) -> T:
        """Returns the Success value, or otherwise calls the failure_function with the exception.

        Parameters
        ----------
        failure_function: Callable[[Exception], T]
            Function which will be called if the Try Monad is Failure.

        Returns
        -------
        result: T
            Returns the Success value or the failure_function result.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.unwrap_or_else(lambda e: len(str(e)))
        4
        """
        match t := self._as_try():
            case Success(value):
                return value
            case Failure(exception):
                return failure_function(exception)
            case _:
                assert_never(t)

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        """Returns the Failure exception, or otherwise a provided default exception.

        Parameters
        ----------
        default_value: Exception
            Default exception

        Returns
        -------
        result: Exception
            Returns the Failure exception or the default value.

        Examples
        --------
        >>> val: Try[int] = Success(1)
        >>> val.unwrap_failure_or(ValueError("default"))
        ValueError('default')
        """
        match t := self._as_try():
            case Success():
                return default_value
            case Failure(exception):
                return exception
            case _:
                assert_never(t)

    def inspect(self, function: Callable[[T], None]) -> Try[T]:
        """Calls function with the Success value (if any) and returns the Try Monad unchanged.

        Parameters
        ----------
        function: Callable[[T], None]
            Inspection function which takes the Success value of the Try Monad

        Returns
        -------
        try: Try[T]

        Examples
        --------
        >>> val: Try[int] = Success(42)
        >>> val.inspect(lambda x: print(f"Value is: {x}"))
        Value is: 42
        Success(42)
        """
        match t := self._as_try():
            case Success(value):
                function(value)
            case Failure():
                pass
            case _:
                assert_never(t)
        return t

    def inspect_failure(self, function: Callable[[Exception], None]) -> Try[T]:
        """Calls function with the Failure exception (if any) and returns the Try Monad unchanged.

        Parameters
        ----------
        function: Callable[[Exception], None]
            Inspection function which takes the exception of the Try Monad

        Returns
        -------
        try: Try[T]

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.inspect_failure(lambda e: print(f"Error: {e}"))
        Error: boom
        Failure(boom)
        """
        match t := self._as_try():
            case Success():
                pass
            case Failure(exception):
                function(exception)
            case _:
                assert_never(t)
        return t

    def match(
        self,
        failure_function: Callable[[Exception], U],
        success_function: Callable[[T], U],
    ) -> U:
        """Matches the Try Monad to either a Failure function or a Success function with the same
        return type.

        Parameters
        ----------
        failure_function: Callable[[Exception], U]
            Callback function for Try monads of type Failure
        success_function: Callable[[T], U]
            Callback function for Try monads of type Success

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.match(lambda e: "Error", lambda x: f"Success: {x}")
        'Success: 10'
        """
        match t := self._as_try():
            case Success(value):
                return success_function(value)
            case Failure(exception):
                return failure_function(exception)
            case _:
                assert_never(t)

    def to_either(self) -> Either[Exception, T]:
        """Converts the Try Monad into an Either Monad (Success -> Right, Failure -> Left).

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.to_either()
        Right(10)
        """
        match t := self._as_try():
            case Success(value):
                return Right(value)
            case Failure(exception):
                return Left(exception)
            case _:
                assert_never(t)

    def to_result(self) -> Result[T, Exception]:
        """Converts the Try Monad into a Result Monad (Success -> Ok, Failure -> Err).

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.to_result()
        Ok(10)
        """
        match t := self._as_try():
            case Success(value):
                return Ok(value)
            case Failure(exception):
                return Err(exception)
            case _:
                assert_never(t)

    def is_success(self) -> bool:
        """Returns True if the Try Monad is Success, otherwise False.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.is_success()
        True
        """
        return isinstance(self, Success)

    def is_failure(self) -> bool:
        """Returns True if the Try Monad is Failure, otherwise False.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.is_failure()
        False
        """
        return isinstance(self, Failure)

    def _payload(self) -> object:
        match t := self._as_try():
            case Success(value):
                return value
            case Failure(exception):
                return exception
            case _:
                assert_never(t)

    def __str__(self) -> str:
        """Returns the string representation of the Try Monad.

        Examples
        --------
        >>> str(Success(42))
        'Success(42)'
        >>> str(Failure(ValueError("boom")))
        'Failure(boom)'
        """
        return f"{type(self).__name__}({self._payload()})"

    def __repr__(self) -> str:
        """Returns the string representation of the Try Monad (same as __str__)."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Returns True if `other` is the same variant wrapping a value of the same type and string,
        so wrapped exceptions compare by type and message.

        Examples
        --------
        >>> Success(1) == Success(1)
        True
        >>> Failure(ValueError("boom")) == Failure(ValueError("boom"))
        True
        >>> Failure(ValueError("boom")) == Failure(TypeError("boom"))
        False
        """
        if type(self) is not type(other):
            return False
        value, other_value = self._payload(), cast("_TryImpl[Any]", other)._payload()
        return type(value) is type(other_value) and str(value) == str(other_value)

    __hash__ = None  # type: ignore[assignment]


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Success(_TryImpl[T]):
    """The Success variant of the Try Monad, wrapping the value of a computation that completed
    without raising.

    Examples
    --------
    >>> Success(42)
    Success(42)
    """

    value: T


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Failure(_TryImpl[T]):
    """The Failure variant of the Try Monad, wrapping the exception a computation raised.

    Examples
    --------
    >>> Failure(ValueError("boom"))
    Failure(boom)
    """

    exception: Exception

    def __post_init__(self) -> None:
        assert isinstance(self.exception, Exception), (
            "Failure value must be of type Exception"
        )


# Failure's success type has deliberately no default: if it defaulted to Never, pyright would pin a
# lambda's success type to Never as soon as one branch returns a Failure
# (`lambda x: Success(x) if x else Failure(ValueError("zero"))`).
type Try[SuccessT] = Success[SuccessT] | Failure[SuccessT]


def safe(function: Callable[[], T]) -> Try[T]:
    """Calls function and wraps its return value in Success, or a raised Exception in Failure.

    Parameters
    ----------
    function: Callable[[], T]
        Callable function which may raise an exception

    Returns
    -------
    try: Try[T]
        Returns a Try Monad which contains either the function result or the raised Exception.

    Examples
    --------
    >>> def risky_call():
    ...     raise ValueError("Boom")
    >>> safe(risky_call)
    Failure(Boom)
    >>> def safe_call():
    ...     return 42
    >>> safe(safe_call)
    Success(42)
    """
    try:
        return Success(function())
    except Exception as e:
        return Failure(e)
