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

The type parameter is covariant and a bare `Failure(e)` is a `Failure[Never]`, so it is assignable
to every `Try[T]` and a conditional lambda such as
`lambda x: Success(x) if x > 0 else Failure(ValueError("negative"))` is inferred as a `Try[int]`.

Two Failures are equal if their exceptions have the same type and `args`, e.g.
`Failure(ValueError("x")) == Failure(ValueError("x"))`.

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
    .match(success=print, failure=lambda e: print(f"Error: {e}")))
```

Structural pattern matching provides a clean, declarative way to handle the contents of a `Try`
monad. `Success` and `Failure` are the only variants (both are `@final`), so a `match` over both is
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

`Try` is a type alias, so use `is_try` (or `isinstance(x, TRY_TYPES)`) for runtime checks and
`is_success`/`is_failure` to narrow a `Try` to one of its variants. Every common method also
exists as a curried module-level function for use with `pymoliath.util.flow`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Never, Self, final, overload

from typing_extensions import Generic, TypeIs, TypeVar

from pymoliath.either import Either, Left, Right
from pymoliath.result import Err, Ok, Result

# Covariant, so `Success[bool]` is a `Try[int]` and a bare `Failure(e)` (a `Failure[Never]`) is
# every `Try[T]`. Like in Rust the receiver fixes the types some methods accept
# (`unwrap_or(default: T)`), which puts the covariant parameter in an input position. That is sound
# here: the containers are immutable and those arguments are only ever returned, typed by the
# receiver's (wider) view. Those methods carry a `# type: ignore[misc]`.
T = TypeVar("T", covariant=True)

# Failure's success type defaults to Never (PEP 696).
T_Never = TypeVar("T_Never", covariant=True, default=Never)

# Method/function-scoped type variables.
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
Y = TypeVar("Y")


class _TryImpl(Generic[T]):
    """Public interface of the Try Monad. The behaviour lives in `Success` and `Failure`, its only
    subclasses."""

    __slots__ = ()

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
        Failure(ValueError('boom'))
        """
        raise NotImplementedError

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
        Failure(ValueError('wrapped boom'))
        """
        raise NotImplementedError

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
        Failure(ValueError('too big'))
        """
        raise NotImplementedError

    def and_then(self, function: Callable[[T], Try[U]]) -> Try[U]:
        """Alias of `bind`, named like in Rust.

        Examples
        --------
        >>> Success(5).and_then(lambda x: Success(x * 2))
        Success(10)
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def apply(self, function: Try[Callable[[T], U]]) -> Try[U]:
        """Applies the function wrapped in `function` to the Success value if both are Success.
        Any exception raised by the function is caught and turned into a Failure.

        If both are Failure, the Failure of `function` takes precedence. For functions of several
        arguments, use `map2`/`map3`.

        Parameters
        ----------
        function: Try[Callable[[T], U]]
            Try Monad which contains a function.

        Returns
        -------
        try: Try[U]
            Returns Success of the function result if both are Success, otherwise a Failure.

        Examples
        --------
        >>> func: Try[Callable[[int], int]] = Success(lambda x: x * 2)
        >>> val: Try[int] = Success(10)
        >>> val.apply(func)
        Success(20)
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def map_or_else(
        self, default_function: Callable[[Exception], U], function: Callable[[T], U]
    ) -> U:
        """Applies the function to the Success value, or the default function to the exception.

        Parameters
        ----------
        default_function: Callable[[Exception], U]
            Function applied to the Failure exception.
        function: Callable[[T], U]
            Function applied to the Success value.

        Returns
        -------
        result: U
            Returns the result of whichever function was applied.

        Examples
        --------
        >>> val: Try[int] = Failure(ValueError("boom"))
        >>> val.map_or_else(lambda e: str(e), lambda x: str(x * 2))
        'boom'
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def flatten(self: _TryImpl[Try[U]]) -> Try[U]:
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
        raise NotImplementedError

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
        raise NotImplementedError

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def inspect(self, function: Callable[[T], None]) -> Self:
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
        raise NotImplementedError

    def inspect_failure(self, function: Callable[[Exception], None]) -> Self:
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
        Failure(ValueError('boom'))
        """
        raise NotImplementedError

    def match(
        self, *, success: Callable[[T], U], failure: Callable[[Exception], U]
    ) -> U:
        """Matches the Try Monad to either the `success` or the `failure` callback with the same
        return type. Both callbacks are keyword-only, so they cannot be swapped by mistake.

        Parameters
        ----------
        success: Callable[[T], U]
            Callback function for Try monads of type Success
        failure: Callable[[Exception], U]
            Callback function for Try monads of type Failure

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.match(success=lambda x: f"Success: {x}", failure=lambda e: "Error")
        'Success: 10'
        """
        raise NotImplementedError

    def to_either(self) -> Either[Exception, T]:
        """Converts the Try Monad into an Either Monad (Success -> Right, Failure -> Left).

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.to_either()
        Right(10)
        """
        raise NotImplementedError

    def to_result(self) -> Result[T, Exception]:
        """Converts the Try Monad into a Result Monad (Success -> Ok, Failure -> Err).

        Examples
        --------
        >>> val: Try[int] = Success(10)
        >>> val.to_result()
        Ok(10)
        """
        raise NotImplementedError

    def is_success(self) -> bool:
        """Returns True if the Try Monad is Success, otherwise False. Use the module-level
        `is_success` to narrow the type.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.is_success()
        True
        """
        raise NotImplementedError

    def is_failure(self) -> bool:
        """Returns True if the Try Monad is Failure, otherwise False. Use the module-level
        `is_failure` to narrow the type.

        Examples
        --------
        >>> val: Try[int] = Success(5)
        >>> val.is_failure()
        False
        """
        raise NotImplementedError


@final
@dataclass(frozen=True, slots=True, repr=False)
class Success(_TryImpl[T]):
    """The Success variant of the Try Monad, wrapping the value of a computation that completed
    without raising.

    Equality and hashing compare the wrapped value (an unhashable value makes the Success
    unhashable).

    Examples
    --------
    >>> Success(42)
    Success(42)
    >>> str(Success("text"))
    'Success(text)'
    """

    value: T

    def map(self, function: Callable[[T], U]) -> Try[U]:
        try:
            return Success(function(self.value))
        except Exception as e:
            return Failure(e)

    def map_failure(self, function: Callable[[Exception], Exception]) -> Self:
        return self

    def bind(self, function: Callable[[T], Try[U]]) -> Try[U]:
        try:
            return function(self.value)
        except Exception as e:
            return Failure(e)

    and_then = bind

    def bind_failure(self, function: Callable[[Exception], Try[T]]) -> Self:
        return self

    def apply(self, function: Try[Callable[[T], U]]) -> Try[U]:
        if isinstance(function, Success):
            try:
                return Success(function.value(self.value))
            except Exception as e:
                return Failure(e)
        return function  # type: ignore[return-value]

    def is_success_and(self, function: Callable[[T], bool]) -> bool:
        return function(self.value)

    def is_failure_and(self, function: Callable[[Exception], bool]) -> bool:
        return False

    def map_or(self, default_value: U, function: Callable[[T], U]) -> U:
        return function(self.value)

    def map_or_else(
        self, default_function: Callable[[Exception], U], function: Callable[[T], U]
    ) -> U:
        return function(self.value)

    def and_(self, other: Try[U]) -> Try[U]:
        return other

    def or_(self, other: Try[T]) -> Self:
        return self

    def zip(self, other: Try[U]) -> Try[tuple[T, U]]:
        if isinstance(other, Success):
            return Success((self.value, other.value))
        return other  # type: ignore[return-value]

    def flatten(self: Success[Try[U]]) -> Try[U]:
        return self.value

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default_value: T) -> T:  # type: ignore[misc]
        return self.value

    def unwrap_or_else(self, failure_function: Callable[[Exception], T]) -> T:
        return self.value

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        return default_value

    def inspect(self, function: Callable[[T], None]) -> Self:
        function(self.value)
        return self

    def inspect_failure(self, function: Callable[[Exception], None]) -> Self:
        return self

    def match(
        self, *, success: Callable[[T], U], failure: Callable[[Exception], U]
    ) -> U:
        return success(self.value)

    def to_either(self) -> Either[Exception, T]:
        return Right(self.value)

    def to_result(self) -> Result[T, Exception]:
        return Ok(self.value)

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"Success({self.value})"

    def __repr__(self) -> str:
        return f"Success({self.value!r})"


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Failure(_TryImpl[T_Never]):
    """The Failure variant of the Try Monad, wrapping the exception a computation raised.

    Two Failures are equal (and hash equally) if their exceptions have the same type and `args`.

    Examples
    --------
    >>> Failure(ValueError("boom"))
    Failure(ValueError('boom'))
    >>> str(Failure(ValueError("boom")))
    'Failure(boom)'
    >>> Failure(ValueError("boom")) == Failure(ValueError("boom"))
    True
    >>> Failure(ValueError("boom")) == Failure(TypeError("boom"))
    False
    """

    exception: Exception

    def __post_init__(self) -> None:
        if not isinstance(self.exception, Exception):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("Failure value must be of type Exception")

    def map(self, function: Callable[[T_Never], U]) -> Failure[U]:
        return self  # type: ignore[return-value]

    def map_failure(
        self, function: Callable[[Exception], Exception]
    ) -> Failure[T_Never]:
        try:
            return Failure(function(self.exception))
        except Exception as e:
            return Failure(e)

    def bind(self, function: Callable[[T_Never], Try[U]]) -> Failure[U]:
        return self  # type: ignore[return-value]

    and_then = bind

    def bind_failure(
        self, function: Callable[[Exception], Try[T_Never]]
    ) -> Try[T_Never]:
        try:
            return function(self.exception)
        except Exception as e:
            return Failure(e)

    def apply(self, function: Try[Callable[[T_Never], U]]) -> Failure[U]:
        if isinstance(function, Failure):
            return function  # type: ignore[return-value]
        return self  # type: ignore[return-value]

    def is_success_and(self, function: Callable[[T_Never], bool]) -> bool:
        return False

    def is_failure_and(self, function: Callable[[Exception], bool]) -> bool:
        return function(self.exception)

    def map_or(self, default_value: U, function: Callable[[T_Never], U]) -> U:
        return default_value

    def map_or_else(
        self,
        default_function: Callable[[Exception], U],
        function: Callable[[T_Never], U],
    ) -> U:
        return default_function(self.exception)

    def and_(self, other: Try[U]) -> Failure[U]:
        return self  # type: ignore[return-value]

    def or_(self, other: Try[T_Never]) -> Try[T_Never]:
        return other

    def zip(self, other: Try[U]) -> Failure[tuple[T_Never, U]]:
        return self  # type: ignore[return-value]

    def flatten(self: Failure[Try[U]]) -> Failure[U]:
        return self  # type: ignore[return-value]

    def unwrap(self) -> Never:
        raise self.exception

    def unwrap_or(self, default_value: T_Never) -> T_Never:  # type: ignore[misc]
        return default_value

    def unwrap_or_else(
        self, failure_function: Callable[[Exception], T_Never]
    ) -> T_Never:
        return failure_function(self.exception)

    def unwrap_failure_or(self, default_value: Exception) -> Exception:
        return self.exception

    def inspect(self, function: Callable[[T_Never], None]) -> Self:
        return self

    def inspect_failure(self, function: Callable[[Exception], None]) -> Self:
        function(self.exception)
        return self

    def match(
        self, *, success: Callable[[T_Never], U], failure: Callable[[Exception], U]
    ) -> U:
        return failure(self.exception)

    def to_either(self) -> Either[Exception, T_Never]:
        return Left(self.exception)

    def to_result(self) -> Result[T_Never, Exception]:
        return Err(self.exception)

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Failure):
            return NotImplemented
        exception: Exception = other.exception
        return type(self.exception) is type(exception) and (
            self.exception.args == exception.args
        )

    def __hash__(self) -> int:
        return hash((type(self.exception), self.exception.args))

    def __str__(self) -> str:
        return f"Failure({self.exception})"

    def __repr__(self) -> str:
        return f"Failure({self.exception!r})"


type Try[SuccessT] = Success[SuccessT] | Failure[SuccessT]

TRY_TYPES: tuple[type[Success[Any]], type[Failure[Any]]] = (Success, Failure)
"""The runtime classes of `Try`, for `isinstance` checks (`Try` itself is a type alias)."""


def is_try(value: object) -> TypeIs[Try[Any]]:
    """Returns True if `value` is a Success or a Failure.

    Examples
    --------
    >>> is_try(Success(1)), is_try(1)
    (True, False)
    """
    return isinstance(value, TRY_TYPES)


def is_success(attempt: Try[U]) -> TypeIs[Success[U]]:
    """Returns True if the Try Monad is Success, narrowing it to `Success` for type checkers.

    Examples
    --------
    >>> val: Try[int] = Success(1)
    >>> if is_success(val):
    ...     print(val.value)
    1
    """
    return isinstance(attempt, Success)


def is_failure(attempt: Try[U]) -> TypeIs[Failure[U]]:
    """Returns True if the Try Monad is Failure, narrowing it to `Failure` for type checkers.

    Examples
    --------
    >>> val: Try[int] = Failure(ValueError("boom"))
    >>> if is_failure(val):
    ...     print(val.exception)
    boom
    """
    return isinstance(attempt, Failure)


def map2(first: Try[U], second: Try[V], function: Callable[[U, V], W]) -> Try[W]:
    """Applies a two-argument function to the values of two Try Monads if both are Success.
    Any exception raised by the function is caught and turned into a Failure.

    If both are Failure, the first Failure takes precedence.

    Examples
    --------
    >>> map2(Success(1), Success(2), lambda a, b: a + b)
    Success(3)
    >>> map2(Failure(ValueError("first")), Failure(ValueError("second")), lambda a, b: a + b)
    Failure(ValueError('first'))
    """
    if isinstance(first, Failure):
        return first  # type: ignore[return-value]
    if isinstance(second, Failure):
        return second  # type: ignore[return-value]
    try:
        return Success(function(first.value, second.value))
    except Exception as e:
        return Failure(e)


def map3(
    first: Try[U],
    second: Try[V],
    third: Try[W],
    function: Callable[[U, V, W], Y],
) -> Try[Y]:
    """Applies a three-argument function to the values of three Try Monads if all are Success.
    Any exception raised by the function is caught and turned into a Failure.

    If several are Failure, the first Failure takes precedence.

    Examples
    --------
    >>> map3(Success(1), Success(2), Success(3), lambda a, b, c: a + b + c)
    Success(6)
    """
    if isinstance(first, Failure):
        return first  # type: ignore[return-value]
    if isinstance(second, Failure):
        return second  # type: ignore[return-value]
    if isinstance(third, Failure):
        return third  # type: ignore[return-value]
    try:
        return Success(function(first.value, second.value, third.value))
    except Exception as e:
        return Failure(e)


@overload
def safe(function: Callable[[], U]) -> Try[U]: ...


@overload
def safe(
    function: Callable[[], U], *, exceptions: tuple[type[Exception], ...]
) -> Try[U]: ...


def safe(
    function: Callable[[], U],
    *,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Try[U]:
    """Calls function and wraps its return value in Success, or a raised exception in Failure.

    Parameters
    ----------
    function: Callable[[], U]
        Callable function which may raise an exception
    exceptions: tuple[type[Exception], ...]
        The exception types to catch (default: `Exception`). Any other exception propagates.

    Returns
    -------
    try: Try[U]
        Returns a Try Monad which contains either the function result or the raised Exception.

    Examples
    --------
    >>> def risky_call():
    ...     raise ValueError("Boom")
    >>> safe(risky_call)
    Failure(ValueError('Boom'))
    >>> def safe_call():
    ...     return 42
    >>> safe(safe_call)
    Success(42)
    >>> safe(lambda: int("x"), exceptions=(ValueError,)).is_failure()
    True
    """
    try:
        return Success(function())
    except exceptions as e:
        return Failure(e)


# Curried module-level functions, for point-free pipelines (see `pymoliath.util.flow`).


def map(function: Callable[[U], V]) -> Callable[[Try[U]], Try[V]]:
    """Curried `Try.map`.

    Examples
    --------
    >>> map(lambda x: x + 1)(Success(1))
    Success(2)
    """
    return lambda attempt: attempt.map(function)


def bind(function: Callable[[U], Try[V]]) -> Callable[[Try[U]], Try[V]]:
    """Curried `Try.bind`.

    Examples
    --------
    >>> bind(lambda x: Success(x + 1))(Success(1))
    Success(2)
    """
    return lambda attempt: attempt.bind(function)


def unwrap_or(default_value: U) -> Callable[[Try[U]], U]:
    """Curried `Try.unwrap_or`.

    Examples
    --------
    >>> unwrap_or(0)(Failure(ValueError("boom")))
    0
    """
    return lambda attempt: attempt.unwrap_or(default_value)


def unwrap_or_else(function: Callable[[Exception], U]) -> Callable[[Try[U]], U]:
    """Curried `Try.unwrap_or_else`.

    Examples
    --------
    >>> unwrap_or_else(lambda e: len(str(e)))(Failure(ValueError("boom")))
    4
    """
    return lambda attempt: attempt.unwrap_or_else(function)


def inspect(function: Callable[[U], None]) -> Callable[[Try[U]], Try[U]]:
    """Curried `Try.inspect`.

    Examples
    --------
    >>> inspect(print)(Success(1))
    1
    Success(1)
    """
    return lambda attempt: attempt.inspect(function)


def inspect_failure(
    function: Callable[[Exception], None],
) -> Callable[[Try[U]], Try[U]]:
    """Curried `Try.inspect_failure`.

    Examples
    --------
    >>> inspect_failure(print)(Failure(ValueError("boom")))
    boom
    Failure(ValueError('boom'))
    """
    return lambda attempt: attempt.inspect_failure(function)
